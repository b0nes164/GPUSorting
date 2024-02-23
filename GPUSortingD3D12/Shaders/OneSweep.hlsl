/******************************************************************************
 * GPUSorting
 * OneSweep Implementation
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 * 
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 ******************************************************************************/
//Compiler Defines
//#define KEY_UINT KEY_INT KEY_FLOAT
//#define PAYLOAD_UINT PAYLOAD_INT PAYLOAD_FLOAT
//#define SHOULD_ASCEND
//#define SORT_PAIRS
//#define ENABLE_16_BIT

//General macros
#if defined(SORT_PAIRS)
    #define PART_SIZE       7680U   //size of a partition tile
    #define PASS_DIM        512U    //The number of threads int digit binning pass
#else
    #define PART_SIZE       3840U
    #define PASS_DIM        256U
#endif
#define G_HIST_DIM          128U    //The number of threads in a global hist threadblock

#define RADIX               256U    //Number of digit bins
#define RADIX_MASK          255U    //Mask of digit bins
#define RADIX_LOG           8U      //log2(RADIX)
#define RADIX_PASSES        4U      //(Key width) / RADIX_LOG

#define HALF_RADIX          128U    //For smaller waves where bit packing is necessary
#define HALF_MASK           127U    // '' 

#define SEC_RADIX_START     256     //Offset for retrieving value from global histogram buffer
#define THIRD_RADIX_START   512     //Offset for retrieving value from global histogram buffer
#define FOURTH_RADIX_START  768     //Offset for retrieving value from global histogram buffer

//For the DigitBinningPass kernel
#define KEYS_PER_THREAD     15U     //The number of keys per thread in a DigitBinningPass threadblock
#if defined(SORT_PAIRS)
    #define MAX_PASS_SMEM   8192U   //shared memory for DigitBinningPass kernel
#else
    #define MAX_PASS_SMEM   4096U   //shared memory for DigitBinningPass kernel
#endif

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3       //Mask used to retrieve flag values

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_radixShift;
    uint e_threadBlocks;
    uint padding;
};

#if defined(KEY_UINT)
RWStructuredBuffer<uint> b_sort         : register(u0);
RWStructuredBuffer<uint> b_alt          : register(u1);
#elif defined(KEY_INT)
RWStructuredBuffer<int> b_sort          : register(u0);
RWStructuredBuffer<int> b_alt           : register(u1);
#elif defined(KEY_FLOAT)
RWStructuredBuffer<float> b_sort        : register(u0);
RWStructuredBuffer<float> b_alt         : register(u1);
#endif

#if defined(PAYLOAD_UINT)
RWStructuredBuffer<uint> b_sortPayload  : register(u2);
RWStructuredBuffer<uint> b_altPayload   : register(u3);
#elif defined(PAYLOAD_INT)
RWStructuredBuffer<int> b_sortPayload   : register(u2);
RWStructuredBuffer<int> b_altPayload    : register(u3);
#elif defined(PAYLOAD_FLOAT)
RWStructuredBuffer<float> b_sortPayload : register(u2);
RWStructuredBuffer<float> b_altPayload  : register(u3);
#endif

RWStructuredBuffer<uint> b_globalHist                   : register(u4); //buffer holding device level offsets for each binning pass
globallycoherent RWStructuredBuffer<uint> b_passHist    : register(u5); //buffer used to store reduced sums of partition tiles
globallycoherent RWStructuredBuffer<uint> b_index       : register(u6); //buffer used to atomically assign partition tile indexes

groupshared uint4 g_gHist[RADIX * 2];   //Shared memory for GlobalHistogram
groupshared uint g_scan[RADIX];         //Shared memory for Scan  
groupshared uint g_pass[MAX_PASS_SMEM]; //Shared memory for DigitBinningPass

inline uint getWaveIndex(uint gtid)
{
    return gtid / WaveGetLaneCount();
}

//Radix Tricks by Michael Herf
//http://stereopsis.com/radix.html
inline uint FloatToUint(float f)
{
    uint mask = -((int) (asuint(f) >> 31)) | 0x80000000;
    return asuint(f) ^ mask;
}

inline float UintToFloat(uint u)
{
    uint mask = ((u >> 31) - 1) | 0x80000000;
    return asfloat(u ^ mask);
}

inline uint IntToUint(int i)
{
    return asuint(i ^ 0x80000000);
}

inline int UintToInt(uint u)
{
    return asint(u ^ 0x80000000);
}

inline uint getWaveCountPass()
{
    return PASS_DIM / WaveGetLaneCount();
}

inline uint ExtractDigit(uint key)
{
    return key >> e_radixShift & RADIX_MASK;
}

inline uint ExtractDigit(uint key, uint shift)
{
    return key >> shift & RADIX_MASK;
}

inline uint ExtractPackedIndex(uint key)
{
    return key >> (e_radixShift + 1) & HALF_MASK;
}

inline uint ExtractPackedShift(uint key)
{
    return (key >> e_radixShift & 1) ? 16 : 0;
}

inline uint ExtractPackedValue(uint packed, uint key)
{
    return packed >> ExtractPackedShift(key) & 0xffff;
}

inline uint SubPartSizeWGE16()
{
    return KEYS_PER_THREAD * WaveGetLaneCount();
}

inline uint SharedOffsetWGE16(uint gtid)
{
    return WaveGetLaneIndex() + getWaveIndex(gtid) * SubPartSizeWGE16();
}

inline uint DeviceOffsetWGE16(uint gtid, uint gid)
{
    return SharedOffsetWGE16(gtid) + gid * PART_SIZE;
}

inline uint SubPartSizeWLT16(uint _serialIterations)
{
    return KEYS_PER_THREAD * WaveGetLaneCount() * _serialIterations;
}

inline uint SharedOffsetWLT16(uint gtid, uint _serialIterations)
{
    return WaveGetLaneIndex() +
        (getWaveIndex(gtid) / _serialIterations * SubPartSizeWLT16(_serialIterations)) +
        (getWaveIndex(gtid) % _serialIterations * WaveGetLaneCount());
}

inline uint DeviceOffsetWLT16(uint gtid, uint gid, uint _serialIterations)
{
    return SharedOffsetWLT16(gtid, _serialIterations) + gid * PART_SIZE;
}

inline uint GlobalHistOffset()
{
    return e_radixShift << 5;
}

inline uint CurrentPass()
{
    return e_radixShift >> 3;
}

inline uint PassHistOffset(uint index)
{
    return ((CurrentPass() * e_threadBlocks) + index) << RADIX_LOG;
}

inline uint WaveHistsSizeWGE16()
{
    return PASS_DIM / WaveGetLaneCount() * RADIX;
}

inline uint WaveHistsSizeWLT16()
{
    return MAX_PASS_SMEM;
}

[numthreads(256, 1, 1)]
void InitOneSweep(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    const uint clearEnd = e_threadBlocks * RADIX * RADIX_PASSES;
    for (uint i = id.x; i < clearEnd; i += increment)
        b_passHist[i] = 0;

    if(id.x < RADIX * RADIX_PASSES)
        b_globalHist[id.x] = 0;
    
    if (id.x < RADIX_PASSES)
        b_index[id.x] = 0;
}

[numthreads(G_HIST_DIM, 1, 1)]
void GlobalHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    //clear shared memory
    const uint histsEnd = RADIX * 2;
    for (uint i = gtid.x; i < histsEnd; i += G_HIST_DIM)
        g_gHist[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    
    //histogram, 64 threads to a histogram
    const uint histOffset = gtid.x / 64 * RADIX;
    const uint partitionEnd = gid.x == e_threadBlocks - 1 ?
        e_numKeys : (gid.x + 1) * PART_SIZE;
    
    uint t;
    for (uint i = gtid.x + gid.x * PART_SIZE; i < partitionEnd; i += G_HIST_DIM)
    {
#if defined(KEY_UINT)
        t = b_sort[i];
#elif defined(KEY_INT)
        t = IntToUint(b_sort[i]);
#elif defined(KEY_FLOAT)
        t = FloatToUint(b_sort[i]);
#endif
        InterlockedAdd(g_gHist[ExtractDigit(t, 0) + histOffset].x, 1);
        InterlockedAdd(g_gHist[ExtractDigit(t, 8) + histOffset].y, 1);
        InterlockedAdd(g_gHist[ExtractDigit(t, 16) + histOffset].z, 1);
        InterlockedAdd(g_gHist[ExtractDigit(t, 24) + histOffset].w, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    
    //reduce counts and add to device
    for (uint i = gtid.x; i < RADIX; i += G_HIST_DIM)
    {
        InterlockedAdd(b_globalHist[i], g_gHist[i].x + g_gHist[i + RADIX].x);
        InterlockedAdd(b_globalHist[i + SEC_RADIX_START], g_gHist[i].y + g_gHist[i + RADIX].y);
        InterlockedAdd(b_globalHist[i + THIRD_RADIX_START], g_gHist[i].z + g_gHist[i + RADIX].z);
        InterlockedAdd(b_globalHist[i + FOURTH_RADIX_START], g_gHist[i].w + g_gHist[i + RADIX].w);
    }
}

[numthreads(RADIX, 1, 1)]
void Scan(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    g_scan[gtid.x] = b_globalHist[gtid.x + gid.x * RADIX];
    g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
    
    //waves 16 or greater can perform a more elegant scan because 16 * 16 = 256
    if (WaveGetLaneCount() >= 16)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x < (RADIX / WaveGetLaneCount()))
        {
            g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint laneMask = WaveGetLaneCount() - 1;
        const uint index = (WaveGetLaneIndex() + 1 & laneMask) + (gtid.x & ~laneMask);
        b_passHist[index + gid.x * RADIX * e_threadBlocks] =
            ((WaveGetLaneIndex() != laneMask ? g_scan[gtid.x] : 0) +
            (gtid.x >= WaveGetLaneCount() ? WaveReadLaneAt(g_scan[gtid.x - 1], 0) : 0)) << 2 | FLAG_INCLUSIVE;
    }
    
    //exclusive Brent-Kung with fused upsweep downsweep
    if (WaveGetLaneCount() < 16)
    {
        const uint passHistOffset = gid.x * RADIX * e_threadBlocks;
        if (gtid.x < WaveGetLaneCount())
        {
            const uint circularLaneShift = WaveGetLaneIndex() + 1 &
                WaveGetLaneCount() - 1;
            b_passHist[circularLaneShift + passHistOffset] =
                (circularLaneShift ? g_scan[gtid.x] : 0) << 2 | FLAG_INCLUSIVE;
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint laneLog = countbits(WaveGetLaneCount() - 1);
        uint offset = laneLog;
        uint j = WaveGetLaneCount();
        for (; j < (RADIX >> 1); j <<= laneLog)
        {
            if(gtid.x < (RADIX >> offset))
            {
                g_scan[((gtid.x + 1) << offset) - 1] +=
                    WavePrefixSum(g_scan[((gtid.x + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
            
            if ((gtid.x & ((j << laneLog) - 1)) >= j)
            {
                if (gtid.x < (j << laneLog))
                {
                    b_passHist[gtid.x + passHistOffset] =
                        (WaveReadLaneAt(g_scan[((gtid.x >> offset) << offset) - 1], 0) +
                        ((gtid.x & (j - 1)) ? g_scan[gtid.x - 1] : 0)) << 2 | FLAG_INCLUSIVE;
                }
                else
                {
                    if ((gtid.x + 1) & (j - 1))
                    {
                        g_scan[gtid.x] +=
                            WaveReadLaneAt(g_scan[((gtid.x >> offset) << offset) - 1], 0);
                    }
                }
            }
            offset += laneLog;
        }
        GroupMemoryBarrierWithGroupSync();
        
        //If RADIX is not an exponentiation of lanecount
        const uint index = gtid.x + j;
        if(index < RADIX)
        {
            b_passHist[index + passHistOffset] =
                (WaveReadLaneAt(g_scan[((index >> offset) << offset) - 1], 0) +
                ((index & (j - 1)) ? g_scan[index - 1] : 0)) << 2 | FLAG_INCLUSIVE;
        }
    }
}

[numthreads(PASS_DIM, 1, 1)]
void DigitBinningPass(uint3 gtid : SV_GroupThreadID)
{
    uint partitionIndex;
    
    //Atomically assign partition tiles
    //WGT16 can clear shared memory early and does not need an extra barrier
    if (WaveGetLaneCount() > 16)
    {
        const uint histsEnd = WaveHistsSizeWGE16();
        for (uint i = gtid.x; i < histsEnd; i += PASS_DIM)
            g_pass[i] = 0;
        
        if (gtid.x == 0)
            InterlockedAdd(b_index[CurrentPass()], 1, g_pass[PART_SIZE - 1]);
        GroupMemoryBarrierWithGroupSync();
        partitionIndex = g_pass[PART_SIZE - 1];
    }
    
    if (WaveGetLaneCount() <= 16)
    {
        if (gtid.x == 0)
            InterlockedAdd(b_index[CurrentPass()], 1, g_pass[0]);
        GroupMemoryBarrierWithGroupSync();
        partitionIndex = g_pass[0];
        GroupMemoryBarrierWithGroupSync();
        
        for (uint i = gtid.x; i < MAX_PASS_SMEM; i += PASS_DIM)
            g_pass[i] = 0;
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (partitionIndex < e_threadBlocks - 1)
    {
        uint keys[KEYS_PER_THREAD];
#if defined(ENABLE_16_BIT)
    uint16_t offsets[KEYS_PER_THREAD];
#else
        uint offsets[KEYS_PER_THREAD];
#endif
        
        if (WaveGetLaneCount() >= 16)
        {
            //Load keys into registers
            [unroll]
            for (uint i = 0, t = DeviceOffsetWGE16(gtid.x, partitionIndex);
                i < KEYS_PER_THREAD;
                ++i, t += WaveGetLaneCount())
            {
#if defined(KEY_UINT)
                keys[i] = b_sort[t];
#elif defined(KEY_INT)
                keys[i] = UintToInt(b_sort[t]);
#elif defined(KEY_FLOAT)
                keys[i] = FloatToUint(b_sort[t]);
#endif
            }
            
            //WLMS
            const uint waveParts = (WaveGetLaneCount() + 31) / 32;
            [unroll]
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            {
                uint4 waveFlags = (WaveGetLaneCount() & 31) ?
                    (1U << WaveGetLaneCount()) - 1 : 0xffffffff;

                [unroll]
                for (uint k = 0; k < RADIX_LOG; ++k)
                {
                    const bool t = keys[i] >> (k + e_radixShift) & 1;
                    const uint4 ballot = WaveActiveBallot(t);
                    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                        waveFlags[wavePart] &= (t ? 0 : 0xffffffff) ^ ballot[wavePart];
                }
                    
                uint bits = 0;
                for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                {
                    if (WaveGetLaneIndex() >= wavePart * 32)
                    {
                        const uint ltMask = WaveGetLaneIndex() >= (wavePart + 1) * 32 ?
                            0xffffffff : (1U << (WaveGetLaneIndex() & 31)) - 1;
                        bits += countbits(waveFlags[wavePart] & ltMask);
                    }
                }
                    
                const uint index = ExtractDigit(keys[i]) + (getWaveIndex(gtid.x) * RADIX);
                offsets[i] = g_pass[index] + bits;
                    
                GroupMemoryBarrierWithGroupSync();
                if (bits == 0)
                {
                    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                        g_pass[index] += countbits(waveFlags[wavePart]);
                }
                GroupMemoryBarrierWithGroupSync();
            }
            
            uint histReduction;
            if (gtid.x < RADIX)
            {
                histReduction = g_pass[gtid.x];
                for (uint i = gtid.x + RADIX; i < WaveHistsSizeWGE16(); i += RADIX)
                {
                    histReduction += g_pass[i];
                    g_pass[i] = histReduction - g_pass[i];
                }
            
                InterlockedAdd(b_passHist[gtid.x + PassHistOffset(partitionIndex + 1)],
                    FLAG_REDUCTION | histReduction << 2);
                histReduction += WavePrefixSum(histReduction);
            }
            GroupMemoryBarrierWithGroupSync();

            if (gtid.x < RADIX)
            {
                const uint laneMask = WaveGetLaneCount() - 1;
                g_pass[((WaveGetLaneIndex() + 1) & laneMask) + (gtid.x & ~laneMask)] = histReduction;
            }
            GroupMemoryBarrierWithGroupSync();
                
            if (gtid.x < RADIX / WaveGetLaneCount())
            {
                g_pass[gtid.x * WaveGetLaneCount()] =
                    WavePrefixSum(g_pass[gtid.x * WaveGetLaneCount()]);
            }
            GroupMemoryBarrierWithGroupSync();
                
            if (gtid.x < RADIX && WaveGetLaneIndex())
                g_pass[gtid.x] += WaveReadLaneAt(g_pass[gtid.x - 1], 1);
            GroupMemoryBarrierWithGroupSync();
            
            //Update offsets
            if (gtid.x >= WaveGetLaneCount())
            {
                const uint t = getWaveIndex(gtid.x) * RADIX;
                [unroll]
                for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                {
                    const uint t2 = ExtractDigit(keys[i]);
                    offsets[i] += g_pass[t2 + t] + g_pass[t2];
                }
            }
            else
            {
                [unroll]
                for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                    offsets[i] += g_pass[ExtractDigit(keys[i])];
            }
            
            uint exclusiveHistReduction;
            if(gtid.x < RADIX)
                exclusiveHistReduction = g_pass[gtid.x];
            GroupMemoryBarrierWithGroupSync();
            
            //Scatter keys
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                g_pass[offsets[i]] = keys[i];

            //Lookback
            if(gtid.x < RADIX)
            {
                uint lookbackReduction = 0;
                for (uint k = partitionIndex; k >= 0;)
                {
                    const uint flagPayload = b_passHist[gtid.x + PassHistOffset(k)];
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        lookbackReduction += flagPayload >> 2;
                        InterlockedAdd(b_passHist[gtid.x + PassHistOffset(partitionIndex + 1)],
                            1 | lookbackReduction << 2);
                        g_pass[gtid.x + PART_SIZE] = lookbackReduction - exclusiveHistReduction;
                        break;
                    }
                    
                    if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
                    {
                        lookbackReduction += flagPayload >> 2;
                        k--;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            
#if defined(SORT_PAIRS)
    #if defined(SHOULD_ASCEND)
            [unroll]
            for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
                 i < KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
            {
                keys[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
        #if defined(KEY_UINT)
                b_alt[keys[i]] = g_pass[t];
        #elif defined(KEY_INT)
                b_alt[keys[i]] = UintToInt(g_pass[t]);
        #elif defined(KEY_FLOAT)
                b_alt[keys[i]] = UintToFloat(g_pass[t]);
        #endif
            }
    #else
            if(e_radixShift == 24)
            {
                [unroll]
                for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
                        i < KEYS_PER_THREAD;
                        ++i, t += WaveGetLaneCount())
                {
                    keys[i] = e_numKeys - g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] - t - 1;
        #if defined(KEY_UINT)
                    b_alt[keys[i]] = g_pass[t];
        #elif defined(KEY_INT)
                    b_alt[keys[i]] = UintToInt(g_pass[t]);
        #elif defined(KEY_FLOAT)
                    b_alt[keys[i]] = UintToFloat(g_pass[t]);
        #endif
                }
            }
            else
            {
                [unroll]
                for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
                        i < KEYS_PER_THREAD;
                        ++i, t += WaveGetLaneCount())
                {
                    keys[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
            #if defined(KEY_UINT)
                    b_alt[keys[i]] = g_pass[t];
            #elif defined(KEY_INT)
                    b_alt[keys[i]] = UintToInt(g_pass[t]);
            #elif defined(KEY_FLOAT)
                    b_alt[keys[i]] = UintToFloat(g_pass[t]);
            #endif
                }
            }
    #endif
            GroupMemoryBarrierWithGroupSync();
                
            [unroll]
            for (uint i = 0, t = DeviceOffsetWGE16(gtid.x, partitionIndex);
                 i < KEYS_PER_THREAD; 
                 ++i, t += WaveGetLaneCount())
            {
    #if defined(PAYLOAD_UINT)
                g_pass[offsets[i]] = b_sortPayload[t];
    #elif defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT)
                g_pass[offsets[i]] = asuint(b_sortPayload[t]);
    #endif
            }
            GroupMemoryBarrierWithGroupSync();
            
            [unroll]
            for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
                 i < KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
            {
    #if defined(PAYLOAD_UINT)
                b_altPayload[keys[i]] = g_pass[t];
    #elif defined(PAYLOAD_INT)
                b_altPayload[keys[i]] = asint(g_pass[t]);
    #elif defined(PAYLOAD_FLOAT)
                b_altPayload[keys[i]] = asfloat(g_pass[t]);
    #endif
            }
#else
    #if defined(SHOULD_ASCEND)
            for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
            {
        #if defined(KEY_UINT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = g_pass[i];
        #elif defined(KEY_INT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToInt(g_pass[i]);
        #elif defined(KEY_FLOAT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToFloat(g_pass[i]);
        #endif
            }
    #else
            if (e_radixShift == 24)
            {
                for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
                {
            #if defined(KEY_UINT)
                b_alt[e_numKeys - g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] - i - 1] = g_pass[i];
            #elif defined(KEY_INT)
                b_alt[e_numKeys - g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] - i - 1] = UintToInt(g_pass[i]);
            #elif defined(KEY_FLOAT)
                b_alt[e_numKeys - g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] - i - 1] = UintToFloat(g_pass[i]);
            #endif
                }
            }
            else
            {
                for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
                {
            #if defined(KEY_UINT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = g_pass[i];
            #elif defined(KEY_INT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToInt(g_pass[i]);
            #elif defined(KEY_FLOAT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToFloat(g_pass[i]);
            #endif
                }
            }
    #endif
#endif
        }
        
        if (WaveGetLaneCount() < 16)
        {
            const uint serialIterations = (PASS_DIM / WaveGetLaneCount() + 31) / 32;
            
            //Load keys into registers
            [unroll]
            for (uint i = 0, t = DeviceOffsetWLT16(gtid.x, partitionIndex, serialIterations);
                i < KEYS_PER_THREAD;
                ++i, t += WaveGetLaneCount() * serialIterations)
            {
#if defined(KEY_UINT)
                keys[i] = b_sort[t];
#elif defined(KEY_INT)
                keys[i] = UintToInt(b_sort[t]);
#elif defined(KEY_FLOAT)
                keys[i] = FloatToUint(b_sort[t]);
#endif
            }
            
            const uint ltMask = (1U << WaveGetLaneIndex()) - 1;
            [unroll]
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            {
                //If full agnostic desired, change 
                uint waveFlag = (1U << WaveGetLaneCount()) - 1;
                
                [unroll]
                for (uint k = 0; k < RADIX_LOG; ++k)
                {
                    const bool t = keys[i] >> (k + e_radixShift) & 1;
                    waveFlag &= (t ? 0 : 0xffffffff) ^ (uint) WaveActiveBallot(t);
                }
                
                uint bits = countbits(waveFlag & ltMask);
                const uint index = ExtractPackedIndex(keys[i]) +
                    (getWaveIndex(gtid.x) / serialIterations * HALF_RADIX);
                    
                for (uint k = 0; k < serialIterations; ++k)
                {
                    if (getWaveIndex(gtid.x) % serialIterations == k)
                        offsets[i] = ExtractPackedValue(g_pass[index], keys[i]) + bits;
                    
                    GroupMemoryBarrierWithGroupSync();
                    if (getWaveIndex(gtid.x) % serialIterations == k && bits == 0)
                    {
                        InterlockedAdd(g_pass[index],
                            countbits(waveFlag) << ExtractPackedShift(keys[i]));
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            }
            
            //inclusive/exclusive prefix sum up the histograms,
            //use a blelloch scan for in place exclusive
            if (gtid.x < HALF_RADIX)
            {
                //If full agnostic desired, change hist
                uint histReduction = g_pass[gtid.x];
                for (uint i = gtid.x + HALF_RADIX; i < WaveHistsSizeWLT16(); i += HALF_RADIX) 
                {
                    histReduction += g_pass[i];
                    g_pass[i] = histReduction - g_pass[i];
                }
                g_pass[gtid.x] = histReduction + (histReduction << 16);
                
                InterlockedAdd(b_passHist[(gtid.x << 1) + PassHistOffset(partitionIndex + 1)],
                    FLAG_REDUCTION | (histReduction & 0xffff) << 2);
                
                InterlockedAdd(b_passHist[(gtid.x << 1) + 1 + PassHistOffset(partitionIndex + 1)],
                    FLAG_REDUCTION | (histReduction >> 16 & 0xffff) << 2);
            }
            
            uint shift = 1;
            for (uint j = RADIX >> 2; j > 0; j >>= 1)
            {
                GroupMemoryBarrierWithGroupSync();
                if(gtid.x < j)
                {
                    g_pass[((((gtid.x << 1) + 2) << shift) - 1) >> 1] +=
                        g_pass[((((gtid.x << 1) + 1) << shift) - 1) >> 1] & 0xffff0000;
                }
                shift++;
            }
            GroupMemoryBarrierWithGroupSync();
                
            if (gtid.x == 0)
                g_pass[HALF_RADIX - 1] &= 0xffff;
                
            for (uint j = 1; j < RADIX >> 1; j <<= 1)
            {
                --shift;
                GroupMemoryBarrierWithGroupSync();
                if(gtid.x < j)
                {
                    const uint t = ((((gtid.x << 1) + 1) << shift) - 1) >> 1;
                    const uint t2 = ((((gtid.x << 1) + 2) << shift) - 1) >> 1;
                    const uint t3 = g_pass[t];
                    g_pass[t] = (g_pass[t] & 0xffff) | (g_pass[t2] & 0xffff0000);
                    g_pass[t2] += t3 & 0xffff0000;
                }
            }

            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < HALF_RADIX)
            {
                const uint t = g_pass[gtid.x];
                g_pass[gtid.x] = (t >> 16) + (t << 16) + (t & 0xffff0000);
            }
            GroupMemoryBarrierWithGroupSync();
            
            //Update offsets
            if (gtid.x >= WaveGetLaneCount() * serialIterations)
            {
                const uint t = getWaveIndex(gtid.x) / serialIterations * HALF_RADIX;
                [unroll]
                for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                {
                    const uint t2 = ExtractPackedIndex(keys[i]);
                    offsets[i] += ExtractPackedValue(g_pass[t2 + t] + g_pass[t2], keys[i]);
                }
            }
            else
            {
                [unroll]
                for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                    offsets[i] += ExtractPackedValue(g_pass[ExtractPackedIndex(keys[i])], keys[i]);
            }
            
            uint exclusiveHistReduction;
            if(gtid.x < RADIX)
                exclusiveHistReduction = g_pass[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
            GroupMemoryBarrierWithGroupSync();
            
            //scatter keys
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                g_pass[offsets[i]] = keys[i];
            
            //Lookback
            if(gtid.x < RADIX)
            {
                uint lookbackReduction = 0;
                for (uint k = partitionIndex; k >= 0;)
                {
                    const uint flagPayload = b_passHist[gtid.x + PassHistOffset(k)];
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        lookbackReduction += flagPayload >> 2;
                        InterlockedAdd(b_passHist[gtid.x + PassHistOffset(partitionIndex + 1)],
                            1 | lookbackReduction << 2);
                        g_pass[gtid.x + PART_SIZE] = lookbackReduction - exclusiveHistReduction;
                        break;
                    }
                    
                    if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
                    {
                        lookbackReduction += flagPayload >> 2;
                        k--;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();

#if defined(SORT_PAIRS)
    #if defined(SHOULD_ASCEND)
            [unroll]
            for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
                 i < KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
            {
                keys[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
        #if defined(KEY_UINT)
                b_alt[keys[i]] = g_pass[t];
        #elif defined(KEY_INT)
                b_alt[keys[i]] = UintToInt(g_pass[t]);
        #elif defined(KEY_FLOAT)
                b_alt[keys[i]] = UintToFloat(g_pass[t]);
        #endif
            }
    #else
            if(e_radixShift == 24)
            {
                [unroll]
                for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
                        i < KEYS_PER_THREAD;
                        ++i, t += WaveGetLaneCount())
                {
                    keys[i] = e_numKeys - g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] - t - 1;
        #if defined(KEY_UINT)
                    b_alt[keys[i]] = g_pass[t];
        #elif defined(KEY_INT)
                    b_alt[keys[i]] = UintToInt(g_pass[t]);
        #elif defined(KEY_FLOAT)
                    b_alt[keys[i]] = UintToFloat(g_pass[t]);
        #endif
                }
            }
            else
            {
                [unroll]
                for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
                        i < KEYS_PER_THREAD;
                        ++i, t += WaveGetLaneCount())
                {
                    keys[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
            #if defined(KEY_UINT)
                    b_alt[keys[i]] = g_pass[t];
            #elif defined(KEY_INT)
                    b_alt[keys[i]] = UintToInt(g_pass[t]);
            #elif defined(KEY_FLOAT)
                    b_alt[keys[i]] = UintToFloat(g_pass[t]);
            #endif
                }
            }
    #endif
            GroupMemoryBarrierWithGroupSync();
                
            [unroll]
            for (uint i = 0, t = DeviceOffsetWLT16(gtid.x, partitionIndex, serialIterations);
                 i < KEYS_PER_THREAD; 
                 ++i, t += WaveGetLaneCount())
            {
    #if defined(PAYLOAD_UINT)
                g_pass[offsets[i]] = b_sortPayload[t];
    #elif defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT)
                g_pass[offsets[i]] = asuint(b_sortPayload[t]);
    #endif
            }
            GroupMemoryBarrierWithGroupSync();
            
            [unroll]
            for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
                 i < KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
            {
    #if defined(PAYLOAD_UINT)
                b_altPayload[keys[i]] = g_pass[t];
    #elif defined(PAYLOAD_INT)
                b_altPayload[keys[i]] = asint(g_pass[t]);
    #elif defined(PAYLOAD_FLOAT)
                b_altPayload[keys[i]] = asfloat(g_pass[t]);
    #endif
            }
#else
    #if defined(SHOULD_ASCEND)
            for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
            {
        #if defined(KEY_UINT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = g_pass[i];
        #elif defined(KEY_INT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToInt(g_pass[i]);
        #elif defined(KEY_FLOAT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToFloat(g_pass[i]);
        #endif
            }
    #else
            if (e_radixShift == 24)
            {
                for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
                {
            #if defined(KEY_UINT)
                b_alt[e_numKeys - g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] - i - 1] = g_pass[i];
            #elif defined(KEY_INT)
                b_alt[e_numKeys - g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] - i - 1] = UintToInt(g_pass[i]);
            #elif defined(KEY_FLOAT)
                b_alt[e_numKeys - g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] - i - 1] = UintToFloat(g_pass[i]);
            #endif
                }
            }
            else
            {
                for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
                {
            #if defined(KEY_UINT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = g_pass[i];
            #elif defined(KEY_INT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToInt(g_pass[i]);
            #elif defined(KEY_FLOAT)
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = UintToFloat(g_pass[i]);
            #endif
                }
            }
    #endif
#endif
        }
    }
    
    //final partition is processed differently
    if(partitionIndex == e_threadBlocks - 1)
    {
        if(gtid.x < RADIX)
        {
            if (partitionIndex)
            {
                uint lookbackReduction = 0;
                for (uint k = partitionIndex; k >= 0;)
                {
                    const uint flagPayload = b_passHist[gtid.x + PassHistOffset(k)];
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        lookbackReduction += flagPayload >> 2;
                        g_pass[gtid.x] = lookbackReduction;
                        break;
                    }
                    
                    if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
                    {
                        lookbackReduction += flagPayload >> 2;
                        k--;
                    }
                }
            }
            else
            {
                g_pass[gtid.x] = b_passHist[gtid.x + PassHistOffset(0)] >> 2;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint waveParts = (WaveGetLaneCount() + 31) / 32;
        const uint partEnd = (e_numKeys + PASS_DIM - 1) / PASS_DIM * PASS_DIM;
        for (uint i = gtid.x + partitionIndex * PART_SIZE; i < partEnd; i += PASS_DIM)
        {
            uint key;
            uint offset;
            uint bits = 0;
            if(i < e_numKeys)
            {
#if defined(KEY_UINT)
                key = b_sort[i];
#elif defined(KEY_INT)
                key = IntToUint(b_sort[i]);
#elif defined(KEY_FLOAT)
                key = FloatToUint(b_sort[i]);
#endif    
            }
            
            uint4 waveFlags = (WaveGetLaneCount() & 31) ?
                (1U << WaveGetLaneCount()) - 1 : 0xffffffff;
            if (i < e_numKeys)
            {
                [unroll]
                for (uint k = 0; k < RADIX_LOG; ++k)
                {
                    const bool t = key >> (k + e_radixShift) & 1;
                    const uint4 ballot = WaveActiveBallot(t);
                    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                        waveFlags[wavePart] &= (t ? 0 : 0xffffffff) ^ ballot[wavePart];
                }
            
                for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                {
                    if (WaveGetLaneIndex() >= wavePart * 32)
                    {
                        const uint ltMask = WaveGetLaneIndex() >= (wavePart + 1) * 32 ?
                            0xffffffff : (1U << (WaveGetLaneIndex() & 31)) - 1;
                        bits += countbits(waveFlags[wavePart] & ltMask);
                    }
                }
            }
            
            for (uint k = 0; k < PASS_DIM / WaveGetLaneCount(); ++k)
            {
                if (getWaveIndex(gtid.x) == k && i < e_numKeys)
                    offset = g_pass[ExtractDigit(key)] + bits;
                GroupMemoryBarrierWithGroupSync();
                
                if (getWaveIndex(gtid.x) == k && i < e_numKeys && bits == 0)
                {
                    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                        g_pass[ExtractDigit(key)] += countbits(waveFlags[wavePart]);
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (i < e_numKeys)
            {
#if defined (SHOULD_ASCEND)
    #if defined(KEY_UINT)
                b_alt[offset] = key;
    #elif defined(KEY_INT)
                b_alt[offset] = UintToInt(key);
    #elif defined(KEY_FLOAT)
                b_alt[offset] = UintToFloat(key);
    #endif
                
    #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
                b_altPayload[offset] = b_sortPayload[i];
    #endif
#else
                if (e_radixShift == 24)
                {
    #if defined(KEY_UINT)
                    b_alt[e_numKeys - offset - 1] = key;
    #elif defined(KEY_INT)
                    b_alt[e_numKeys - offset - 1] = UintToInt(key);
    #elif defined(KEY_FLOAT)
                    b_alt[e_numKeys - offset - 1] = UintToFloat(key);
    #endif
    #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
                    b_altPayload[e_numKeys - offset - 1] = b_sortPayload[i];
    #endif   
                }
                else
                {
    #if defined(KEY_UINT)
                    b_alt[offset] = key;
    #elif defined(KEY_INT)
                    b_alt[offset] = UintToInt(key);
    #elif defined(KEY_FLOAT)
                    b_alt[offset] = UintToFloat(key);
    #endif
    #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
                    b_altPayload[offset] = b_sortPayload[i];
    #endif   
                }
#endif
            }
        }
    }
}