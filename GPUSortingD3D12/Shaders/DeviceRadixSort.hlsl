/******************************************************************************
 * GPUSorting
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
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
    #define PART_SIZE   7680U   //size of a partition tile
    #define DS_DIM      512U    //The number of threads int digit binning pass
#else
    #define PART_SIZE   3840U
    #define DS_DIM      256U
#endif
#define US_DIM          128U    //The number of threads in a Upsweep threadblock
#define SCAN_DIM        128U    //The number of threads in a Scan threadblock

#define RADIX           256U    //Number of digit bins
#define RADIX_MASK      255U    //Mask of digit bins
#define RADIX_LOG       8U      //log2(RADIX)

#define HALF_RADIX      128U    //For smaller waves where bit packing is necessary
#define HALF_MASK       127U    // '' 

//For the downsweep kernels
#define DS_KEYS_PER_THREAD  15U     //The number of keys per thread in a Downsweep Threadblock
#if defined(SORT_PAIRS)
    #define MAX_DS_SMEM   8192U   //shared memory for DigitBinningPass kernel
#else
    #define MAX_DS_SMEM   4096U   //shared memory for DigitBinningPass kernel
#endif

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_radixShift;
    uint e_threadBlocks;
    uint padding;
};

#if defined(KEY_UINT)
RWStructuredBuffer<uint> b_sort         : register(u0);
RWStructuredBuffer<uint> b_alt          : register(u2);
#elif defined(KEY_INT)
RWStructuredBuffer<int> b_sort          : register(u0);
RWStructuredBuffer<int> b_alt           : register(u2);
#elif defined(KEY_FLOAT)
RWStructuredBuffer<float> b_sort        : register(u0);
RWStructuredBuffer<float> b_alt         : register(u2);
#endif

#if defined(PAYLOAD_UINT)
RWStructuredBuffer<uint> b_sortPayload  : register(u1);
RWStructuredBuffer<uint> b_altPayload   : register(u3);
#elif defined(PAYLOAD_INT)
RWStructuredBuffer<int> b_sortPayload   : register(u1);
RWStructuredBuffer<int> b_altPayload    : register(u3);
#elif defined(PAYLOAD_FLOAT)
RWStructuredBuffer<float> b_sortPayload : register(u1);
RWStructuredBuffer<float> b_altPayload  : register(u3);
#endif

RWStructuredBuffer<uint> b_globalHist   : register(u4); //buffer holding device level offsets for each binning pass
RWStructuredBuffer<uint> b_passHist     : register(u5); //buffer used to store reduced sums of partition tiles

groupshared uint g_us[RADIX * 2];   //Shared memory for upsweep
groupshared uint g_scan[SCAN_DIM];  //Shared memory for the scan
groupshared uint g_ds[MAX_DS_SMEM]; //Shared memory for the downsweep

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

inline uint ExtractDigit(uint key)
{
    return key >> e_radixShift & RADIX_MASK;
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
    return DS_KEYS_PER_THREAD * WaveGetLaneCount();
}

inline uint SharedOffsetWGE16(uint gtid)
{
    return WaveGetLaneIndex() + getWaveIndex(gtid) * SubPartSizeWGE16();
}

inline uint SubPartSizeWLT16(uint _serialIterations)
{
    return DS_KEYS_PER_THREAD * WaveGetLaneCount() * _serialIterations;
}

inline uint SharedOffsetWLT16(uint gtid, uint _serialIterations)
{
    return WaveGetLaneIndex() +
        (getWaveIndex(gtid) / _serialIterations * SubPartSizeWLT16(_serialIterations)) +
        (getWaveIndex(gtid) % _serialIterations * WaveGetLaneCount());
}

inline uint SharedOffset(uint gtid, uint _serialIterations)
{
    return (WaveGetLaneCount() < 16 ? 
        SharedOffsetWLT16(gtid, _serialIterations) :
        SharedOffsetWGE16(gtid));
}

inline uint DeviceOffset(uint gtid, uint gid, uint _serialIterations)
{
    return SharedOffset(gtid, _serialIterations) + gid * PART_SIZE;
}

inline uint GlobalHistOffset()
{
    return e_radixShift << 5;
}

inline uint WaveHistsSizeWGE16()
{
    return DS_DIM / WaveGetLaneCount() * RADIX;
}

inline uint WaveHistsSizeWLT16()
{
    return MAX_DS_SMEM;
}

//Clear the global histogram, as we will be adding to it atomically
[numthreads(1024, 1, 1)]
void InitDeviceRadixSort(int3 id : SV_DispatchThreadID)
{
    b_globalHist[id.x] = 0;
}

[numthreads(US_DIM, 1, 1)]
void Upsweep(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    //clear shared memory
    const uint histsEnd = RADIX * 2;
    for (uint i = gtid.x; i < histsEnd; i += US_DIM)
        g_us[i] = 0;
    GroupMemoryBarrierWithGroupSync();

    //histogram, 64 threads to a histogram
    const uint histOffset = gtid.x / 64 * RADIX;
    const uint partitionEnd = gid.x == e_threadBlocks - 1 ?
        e_numKeys : (gid.x + 1) * PART_SIZE;
    for (uint i = gtid.x + gid.x * PART_SIZE; i < partitionEnd; i += US_DIM)
    {
#if defined(KEY_UINT)
        InterlockedAdd(g_us[ExtractDigit(b_sort[i]) + histOffset], 1);
#elif defined(KEY_INT)
        InterlockedAdd(g_us[ExtractDigit(IntToUint(b_sort[i])) + histOffset], 1);
#elif defined(KEY_FLOAT)
        InterlockedAdd(g_us[ExtractDigit(FloatToUint(b_sort[i])) + histOffset], 1);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    
    //reduce and pass to tile histogram
    for (uint i = gtid.x; i < RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + RADIX];
        b_passHist[i * e_threadBlocks + gid.x] = g_us[i];
        g_us[i] += WavePrefixSum(g_us[i]);
    }
    
    //Larger 16 or greater can perform a more elegant scan because 16 * 16 = 256
    if (WaveGetLaneCount() >= 16)
    {
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < (RADIX / WaveGetLaneCount()))
        {
            g_us[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_us[(gtid.x + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        //atomically add to global histogram
        const uint globalHistOffset = GlobalHistOffset();
        const uint laneMask = WaveGetLaneCount() - 1;
        const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
        for (uint i = gtid.x; i < RADIX; i += US_DIM)
        {
            const uint index = circularLaneShift + (i & ~laneMask);
            InterlockedAdd(b_globalHist[index + globalHistOffset],
                (WaveGetLaneIndex() != laneMask ? g_us[i] : 0) +
                (i >= WaveGetLaneCount() ? WaveReadLaneAt(g_us[i - 1], 0) : 0));
        }
    }
    
    //Exclusive Brent-Kung with fused upsweep downsweep
    if (WaveGetLaneCount() < 16)
    {
        const uint globalHistOffset = GlobalHistOffset();
        if (gtid.x < WaveGetLaneCount())
        {
            const uint circularLaneShift = WaveGetLaneIndex() + 1 & 
                WaveGetLaneCount() - 1;
            InterlockedAdd(b_globalHist[circularLaneShift + globalHistOffset],
                circularLaneShift ? g_us[gtid.x] : 0);
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint laneLog = countbits(WaveGetLaneCount() - 1);
        uint offset = laneLog;
        uint j = WaveGetLaneCount();
        for (; j < (RADIX >> 1); j <<= laneLog)
        {
            if (gtid.x < (RADIX >> offset))
            {
                g_us[((gtid.x + 1) << offset) - 1] +=
                    WavePrefixSum(g_us[((gtid.x + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
            
            for (uint i = gtid.x + j; i < RADIX; i += US_DIM)
            {
                if ((i & ((j << laneLog) - 1)) >= j)
                {
                    if (i < (j << laneLog))
                    {
                        InterlockedAdd(b_globalHist[i + globalHistOffset],
                            WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0) +
                            ((i & (j - 1)) ? g_us[i - 1] : 0));
                    }
                    else
                    {
                        if ((i + 1) & (j - 1))
                        {
                            g_us[i] +=
                                WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0);
                        }
                    }
                }
            }
            offset += laneLog;
        }
        GroupMemoryBarrierWithGroupSync();
        
        //If RADIX is not a multiple of lanecount
        for (uint i = gtid.x + j; i < RADIX; i += US_DIM)
        {
            InterlockedAdd(b_globalHist[i + globalHistOffset],
                WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0) +
                ((i & (j - 1)) ? g_us[i - 1] : 0));
        }
    }
}

//Scan along the spine of the upsweep
[numthreads(SCAN_DIM, 1, 1)]
void Scan(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (WaveGetLaneCount() >= 16)
    {
        uint aggregate = 0;
        const uint laneMask = WaveGetLaneCount() - 1;
        const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
        const uint partionsEnd = e_threadBlocks / SCAN_DIM * SCAN_DIM;
        const uint offset = gid.x * e_threadBlocks;
        uint i = gtid.x;
        for (; i < partionsEnd; i += SCAN_DIM)
        {
            g_scan[gtid.x] = b_passHist[i + offset];
            g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
            GroupMemoryBarrierWithGroupSync();
            
            if (gtid.x < SCAN_DIM / WaveGetLaneCount())
            {
                g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                    WavePrefixSum(g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
            
            b_passHist[circularLaneShift + (i & ~laneMask) + offset] =
                (WaveGetLaneIndex() != laneMask ? g_scan[gtid.x] : 0) +
                (gtid.x >= WaveGetLaneCount() ?
                WaveReadLaneAt(g_scan[gtid.x - 1], 0) : 0) +
                aggregate;

            aggregate += g_scan[SCAN_DIM - 1];
            GroupMemoryBarrierWithGroupSync();
        }
        
        //partial
        if (i < e_threadBlocks)
            g_scan[gtid.x] = b_passHist[offset + i];
        g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
        GroupMemoryBarrierWithGroupSync();
            
        if (gtid.x < SCAN_DIM / WaveGetLaneCount())
        {
            g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_scan[(gtid.x + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint index = circularLaneShift + (i & ~laneMask);
        if (index < e_threadBlocks)
        {
            b_passHist[index + offset] = 
                (WaveGetLaneIndex() != laneMask ? g_scan[gtid.x] : 0) +
                (gtid.x >= WaveGetLaneCount() ?
                g_scan[(gtid.x & ~laneMask) - 1] : 0) +
                aggregate;
        }
    }

    if (WaveGetLaneCount() < 16)
    {
        uint aggregate = 0;
        const uint partitions = e_threadBlocks / SCAN_DIM;
        const uint deviceOffset = gid.x * e_threadBlocks;
        const uint laneLog = countbits(WaveGetLaneCount() - 1);
        const uint circularLaneShift = WaveGetLaneIndex() + 1 &
                    WaveGetLaneCount() - 1;
        
        uint k = 0;
        for (; k < partitions; ++k)
        {
            g_scan[gtid.x] = b_passHist[gtid.x + k * SCAN_DIM + deviceOffset];
            g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
            
            if (gtid.x < WaveGetLaneCount())
            {
                b_passHist[circularLaneShift + k * SCAN_DIM + deviceOffset] =
                    (circularLaneShift ? g_scan[gtid.x] : 0) + aggregate;
            }
            GroupMemoryBarrierWithGroupSync();
            
            uint offset = laneLog;
            uint j = WaveGetLaneCount();
            for (; j < (SCAN_DIM >> 1); j <<= laneLog)
            {
                if (gtid.x < (SCAN_DIM >> offset))
                {
                    g_scan[((gtid.x + 1) << offset) - 1] +=
                        WavePrefixSum(g_scan[((gtid.x + 1) << offset) - 1]);
                }
                GroupMemoryBarrierWithGroupSync();
            
                if ((gtid.x & ((j << laneLog) - 1)) >= j)
                {
                    if (gtid.x < (j << laneLog))
                    {
                        b_passHist[gtid.x + k * SCAN_DIM + deviceOffset] =
                            WaveReadLaneAt(g_scan[((gtid.x >> offset) << offset) - 1], 0) +
                            ((gtid.x & (j - 1)) ? g_scan[gtid.x - 1] : 0) + aggregate;
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
        
            //If SCAN_DIM is not a multiple of lanecount
            for (uint i = gtid.x + j; i < SCAN_DIM; i += SCAN_DIM)
            {
                b_passHist[i + k * SCAN_DIM + deviceOffset] =
                    WaveReadLaneAt(g_scan[((i >> offset) << offset) - 1], 0) +
                    ((i & (j - 1)) ? g_scan[i - 1] : 0) + aggregate;
            }
            
            aggregate += WaveReadLaneAt(g_scan[SCAN_DIM - 1], 0) +
                WaveReadLaneAt(g_scan[(((SCAN_DIM - 1) >> offset) << offset) - 1], 0);
            GroupMemoryBarrierWithGroupSync();
        }
        
        //partial
        const uint finalPartSize = e_threadBlocks - k * SCAN_DIM;
        if (gtid.x < finalPartSize)
        {
            g_scan[gtid.x] = b_passHist[gtid.x + k * SCAN_DIM + deviceOffset];
            g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
        }
        
        if (gtid.x < WaveGetLaneCount() && circularLaneShift < finalPartSize)
        {
            b_passHist[circularLaneShift + k * SCAN_DIM + deviceOffset] =
                    (circularLaneShift ? g_scan[gtid.x] : 0) + aggregate;
        }
        GroupMemoryBarrierWithGroupSync();
        
        uint offset = laneLog;
        for (uint j = WaveGetLaneCount(); j < finalPartSize; j <<= laneLog)
        {
            if (gtid.x < (finalPartSize >> offset))
            {
                g_scan[((gtid.x + 1) << offset) - 1] +=
                        WavePrefixSum(g_scan[((gtid.x + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
            
            if ((gtid.x & ((j << laneLog) - 1)) >= j && gtid.x < finalPartSize)
            {
                if (gtid.x < (j << laneLog))
                {
                    b_passHist[gtid.x + k * SCAN_DIM + deviceOffset] =
                        WaveReadLaneAt(g_scan[((gtid.x >> offset) << offset) - 1], 0) +
                        ((gtid.x & (j - 1)) ? g_scan[gtid.x - 1] : 0) + aggregate;
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
    }
}

[numthreads(DS_DIM, 1, 1)]
void Downsweep(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    //If the size of  a wave is too small, we do not have enough space in
    //shared memory to assign a histogram to each wave, so instead,
    //some operations are peformed serially.
    const uint serialIterations = (DS_DIM / WaveGetLaneCount() + 31) / 32;
    
    uint keys[DS_KEYS_PER_THREAD];
#if defined(ENABLE_16_BIT)
        uint16_t offsets[DS_KEYS_PER_THREAD];
#else
    uint offsets[DS_KEYS_PER_THREAD];
#endif
    
    //Load keys into registers
    if (gid.x < e_threadBlocks - 1)
    {
        [unroll]
        for (uint i = 0, t = DeviceOffset(gtid.x, gid.x, serialIterations);
                 i < DS_KEYS_PER_THREAD;
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
    }
    
    //If this is the last threadblock, load dummy keys into registers
    //if we exceed the sorting input. Because dummy key is the highest
    //possible digit, we guaruntee that it will be placed last when
    //scattered into shared memory
    if (gid.x == e_threadBlocks - 1)
    {
        [unroll]
        for (uint i = 0, t = DeviceOffset(gtid.x, gid.x, serialIterations);
                 i < DS_KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount() * serialIterations)
        {
            if(t < e_numKeys)
            {
#if defined(KEY_UINT)
                keys[i] = b_sort[t];
#elif defined(KEY_INT)
                keys[i] = UintToInt(b_sort[t]);
#elif defined(KEY_FLOAT)
                keys[i] = FloatToUint(b_sort[t]);
#endif
            }
            else
            {
                keys[i] = 0xffffffff;
            }
        }
    }
    
    if (WaveGetLaneCount() >= 16)
    {
        //Clear histogram memory
        for (uint i = gtid.x; i < WaveHistsSizeWGE16(); i += DS_DIM)
            g_ds[i] = 0;
        GroupMemoryBarrierWithGroupSync();

        //Warp Level Multisplit
        const uint waveParts = (WaveGetLaneCount() + 31) / 32;
        [unroll]
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
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
            offsets[i] = g_ds[index] + bits;
                    
            GroupMemoryBarrierWithGroupSync();
            if (bits == 0)
            {
                for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                    g_ds[index] += countbits(waveFlags[wavePart]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
            
        //inclusive/exclusive prefix sum up the histograms
        //followed by exclusive prefix sum across the reductions
        uint histReduction;
        if (gtid.x < RADIX)
        {
            histReduction = g_ds[gtid.x];
            for (uint i = gtid.x + RADIX; i < WaveHistsSizeWGE16(); i += RADIX)
            {
                histReduction += g_ds[i];
                g_ds[i] = histReduction - g_ds[i];
            }
            
            histReduction += WavePrefixSum(histReduction);
        }
        GroupMemoryBarrierWithGroupSync();

        if (gtid.x < RADIX)
        {
            const uint laneMask = WaveGetLaneCount() - 1;
            g_ds[((WaveGetLaneIndex() + 1) & laneMask) + (gtid.x & ~laneMask)] = histReduction;
        }
        GroupMemoryBarrierWithGroupSync();
                
        if (gtid.x < RADIX / WaveGetLaneCount())
        {
            g_ds[gtid.x * WaveGetLaneCount()] =
                    WavePrefixSum(g_ds[gtid.x * WaveGetLaneCount()]);
        }
        GroupMemoryBarrierWithGroupSync();
                
        if (gtid.x < RADIX && WaveGetLaneIndex())
            g_ds[gtid.x] += WaveReadLaneAt(g_ds[gtid.x - 1], 1);
        GroupMemoryBarrierWithGroupSync();
        
        //Update offsets
        if (gtid.x >= WaveGetLaneCount())
        {
            const uint t = getWaveIndex(gtid.x) * RADIX;
            [unroll]
            for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
            {
                const uint t2 = ExtractDigit(keys[i]);
                offsets[i] += g_ds[t2 + t] + g_ds[t2];
            }
        }
        else
        {
            [unroll]
            for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
                offsets[i] += g_ds[ExtractDigit(keys[i])];
        }
            
        //take advantage of barrier
        //Note: Don't remove this again LEMAo
        uint exclusiveHistReduction;
        if (gtid.x < RADIX)
            exclusiveHistReduction = g_ds[gtid.x];
        GroupMemoryBarrierWithGroupSync();
            
        //scatter keys into shared memory
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
            g_ds[offsets[i]] = keys[i];
            
        if (gtid.x < RADIX)
        {
            g_ds[gtid.x + PART_SIZE] = b_globalHist[gtid.x + GlobalHistOffset()] +
                    b_passHist[gtid.x * e_threadBlocks + gid.x] - exclusiveHistReduction;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (WaveGetLaneCount() < 16)
    {
        //clear shared memory
        for (uint i = gtid.x; i < WaveHistsSizeWLT16(); i += DS_DIM)
            g_ds[i] = 0;
        GroupMemoryBarrierWithGroupSync();
            
        const uint ltMask = (1U << WaveGetLaneIndex()) - 1;
        [unroll]
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
        {
            uint waveFlag = (1U << WaveGetLaneCount()) - 1; //for full agnostic add ternary and uint4
                
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
                    offsets[i] = ExtractPackedValue(g_ds[index], keys[i]) + bits;
                    
                GroupMemoryBarrierWithGroupSync();
                if (getWaveIndex(gtid.x) % serialIterations == k && bits == 0)
                {
                    InterlockedAdd(g_ds[index],
                            countbits(waveFlag) << ExtractPackedShift(keys[i]));
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
            
        //inclusive/exclusive prefix sum up the histograms,
        //use a blelloch scan for in place exclusive
        uint reduction;
        if (gtid.x < HALF_RADIX)
        {
            reduction = g_ds[gtid.x];
            for (uint i = gtid.x + HALF_RADIX; i < WaveHistsSizeWLT16(); i += HALF_RADIX)
            {
                reduction += g_ds[i];
                g_ds[i] = reduction - g_ds[i];
            }
            g_ds[gtid.x] = reduction + (reduction << 16);
        }
                
        uint shift = 1;
        for (uint j = RADIX >> 2; j > 0; j >>= 1)
        {
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < j)
            {
                g_ds[((((gtid.x << 1) + 2) << shift) - 1) >> 1] +=
                            g_ds[((((gtid.x << 1) + 1) << shift) - 1) >> 1] & 0xffff0000;
            }
            shift++;
        }
        GroupMemoryBarrierWithGroupSync();
                
        if (gtid.x == 0)
            g_ds[HALF_RADIX - 1] &= 0xffff;
                
        for (uint j = 1; j < RADIX >> 1; j <<= 1)
        {
            --shift;
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < j)
            {
                const uint t = ((((gtid.x << 1) + 1) << shift) - 1) >> 1;
                const uint t2 = ((((gtid.x << 1) + 2) << shift) - 1) >> 1;
                const uint t3 = g_ds[t];
                g_ds[t] = (g_ds[t] & 0xffff) | (g_ds[t2] & 0xffff0000);
                g_ds[t2] += t3 & 0xffff0000;
            }
        }

        GroupMemoryBarrierWithGroupSync();
        if (gtid.x < HALF_RADIX)
        {
            const uint t = g_ds[gtid.x];
            g_ds[gtid.x] = (t >> 16) + (t << 16) + (t & 0xffff0000);
        }
        GroupMemoryBarrierWithGroupSync();
            
        //Update offsets
        if (gtid.x >= WaveGetLaneCount() * serialIterations)
        {
            const uint t = getWaveIndex(gtid.x) / serialIterations * HALF_RADIX;
            [unroll]
            for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
            {
                const uint t2 = ExtractPackedIndex(keys[i]);
                offsets[i] += ExtractPackedValue(g_ds[t2 + t] + g_ds[t2], keys[i]);
            }
        }
        else
        {
            [unroll]
            for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
                offsets[i] += ExtractPackedValue(g_ds[ExtractPackedIndex(keys[i])], keys[i]);
        }
            
        uint exclusiveHistReduction;
        if (gtid.x < RADIX)
            exclusiveHistReduction = g_ds[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
        GroupMemoryBarrierWithGroupSync();
            
        //scatter keys into shared memory
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
            g_ds[offsets[i]] = keys[i];
        
        if (gtid.x < RADIX)
        {
            g_ds[gtid.x + PART_SIZE] = b_globalHist[gtid.x + GlobalHistOffset()] +
                    b_passHist[gtid.x * e_threadBlocks + gid.x] - exclusiveHistReduction;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    //Scatter keys into device
    if(gid.x < e_threadBlocks - 1)
    {
#if defined(SORT_PAIRS)
    #if defined(SHOULD_ASCEND)
        [unroll]
        for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                i < DS_KEYS_PER_THREAD;
                ++i, t += WaveGetLaneCount() * serialIterations)
        {
            keys[i] = g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] + t;
        #if defined(KEY_UINT)
            b_alt[keys[i]] = g_ds[t];
        #elif defined(KEY_INT)
            b_alt[keys[i]] = UintToInt(g_ds[t]);
        #elif defined(KEY_FLOAT)
            b_alt[keys[i]] = UintToFloat(g_ds[t]);
        #endif
        }
    #else
        if(e_radixShift == 24)
        {
            [unroll]
            for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                    i < DS_KEYS_PER_THREAD;
                    ++i, t += WaveGetLaneCount() * serialIterations)
            {
                keys[i] = e_numKeys - g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] - t - 1;
        #if defined(KEY_UINT)
                b_alt[keys[i]] = g_ds[t];
        #elif defined(KEY_INT)
                b_alt[keys[i]] = UintToInt(g_ds[t]);
        #elif defined(KEY_FLOAT)
                b_alt[keys[i]] = UintToFloat(g_ds[t]);
        #endif
            }
        }
        else
        {
            [unroll]
            for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                    i < DS_KEYS_PER_THREAD;
                    ++i, t += WaveGetLaneCount() * serialIterations)
            {
                keys[i] = g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] + t;
            #if defined(KEY_UINT)
                b_alt[keys[i]] = g_ds[t];
            #elif defined(KEY_INT)
                b_alt[keys[i]] = UintToInt(g_ds[t]);
            #elif defined(KEY_FLOAT)
                b_alt[keys[i]] = UintToFloat(g_ds[t]);
            #endif
            }
        }
    #endif
        GroupMemoryBarrierWithGroupSync();
                
        [unroll]
        for (uint i = 0, t = DeviceOffset(gtid.x, gid.x, serialIterations);
                i < DS_KEYS_PER_THREAD; 
                ++i, t += WaveGetLaneCount() * serialIterations)
        {
    #if defined(PAYLOAD_UINT)
            g_ds[offsets[i]] = b_sortPayload[t];
    #elif defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT)
            g_ds[offsets[i]] = asuint(b_sortPayload[t]);
    #endif
        }
        GroupMemoryBarrierWithGroupSync();
            
        [unroll]
        for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                i < DS_KEYS_PER_THREAD;
                ++i, t += WaveGetLaneCount() * serialIterations)
        {
    #if defined(PAYLOAD_UINT)
            b_altPayload[keys[i]] = g_ds[t];
    #elif defined(PAYLOAD_INT)
            b_altPayload[keys[i]] = asint(g_ds[t]);
    #elif defined(PAYLOAD_FLOAT)
            b_altPayload[keys[i]] = asfloat(g_ds[t]);
    #endif
        }
#else
    #if defined(SHOULD_ASCEND)
        for (uint i = gtid.x; i < PART_SIZE; i += DS_DIM)
        {
        #if defined(KEY_UINT)
            b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = g_ds[i];
        #elif defined(KEY_INT)
            b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToInt(g_ds[i]);
        #elif defined(KEY_FLOAT)
            b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToFloat(g_ds[i]);
        #endif
        }
    #else
        if (e_radixShift == 24)
        {
            for (uint i = gtid.x; i < PART_SIZE; i += DS_DIM)
            {
        #if defined(KEY_UINT)
                b_alt[e_numKeys - g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] - i - 1] = g_ds[i];
        #elif defined(KEY_INT)
                b_alt[e_numKeys - g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] - i - 1] = UintToInt(g_ds[i]);
        #elif defined(KEY_FLOAT)
                b_alt[e_numKeys - g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] - i - 1] = UintToFloat(g_ds[i]);
        #endif
            }
        }
        else
        {
            for (uint i = gtid.x; i < PART_SIZE; i += DS_DIM)
            {
        #if defined(KEY_UINT)
                b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = g_ds[i];
        #elif defined(KEY_INT)
                b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToInt(g_ds[i]);
        #elif defined(KEY_FLOAT)
                b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToFloat(g_ds[i]);
        #endif
            }
        }
    #endif
#endif
    }
    
    if(gid.x == e_threadBlocks - 1)
    {
#if defined(SORT_PAIRS)
        int subPartSize = (int)(e_numKeys - gid.x * PART_SIZE) - 
            (int)(getWaveIndex(gtid.x) / serialIterations * DS_KEYS_PER_THREAD * WaveGetLaneCount() * serialIterations);
        uint finalKeys;
        if (subPartSize > 0)
        {
            if((uint)subPartSize >= DS_KEYS_PER_THREAD * WaveGetLaneCount() * serialIterations)
            {
                finalKeys = DS_KEYS_PER_THREAD;
            }
            else
            {
                finalKeys = subPartSize / WaveGetLaneCount() / serialIterations;
                subPartSize -= finalKeys * WaveGetLaneCount() * serialIterations;
                if(WaveGetLaneIndex() + getWaveIndex(gtid.x) % serialIterations * WaveGetLaneCount() < (uint)subPartSize)
                    finalKeys++;
            }
        }
        else
        {
            finalKeys = 0;
        }
    #if defined(SHOULD_ASCEND)
        for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                i < finalKeys;
                ++i, t += WaveGetLaneCount() * serialIterations)
        {
            keys[i] = g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] + t;
        #if defined(KEY_UINT)
            b_alt[keys[i]] = g_ds[t];
        #elif defined(KEY_INT)
            b_alt[keys[i]] = UintToInt(g_ds[t]);
        #elif defined(KEY_FLOAT)
            b_alt[keys[i]] = UintToFloat(g_ds[t]);
        #endif
        }
    #else
        if(e_radixShift == 24)
        {
            for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                    i < finalKeys;
                    ++i, t += WaveGetLaneCount() * serialIterations)
            {
                keys[i] = e_numKeys - g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] - t - 1;
        #if defined(KEY_UINT)
                b_alt[keys[i]] = g_ds[t];
        #elif defined(KEY_INT)
                b_alt[keys[i]] = UintToInt(g_ds[t]);
        #elif defined(KEY_FLOAT)
                b_alt[keys[i]] = UintToFloat(g_ds[t]);
        #endif
            }
        }
        else
        {
            for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                    i < finalKeys;
                    ++i, t += WaveGetLaneCount() * serialIterations)
            {
                keys[i] = g_ds[ExtractDigit(g_ds[t]) + PART_SIZE] + t;
            #if defined(KEY_UINT)
                b_alt[keys[i]] = g_ds[t];
            #elif defined(KEY_INT)
                b_alt[keys[i]] = UintToInt(g_ds[t]);
            #elif defined(KEY_FLOAT)
                b_alt[keys[i]] = UintToFloat(g_ds[t]);
            #endif
            }
        }
    #endif
        GroupMemoryBarrierWithGroupSync();
                
        for (uint i = 0, t = DeviceOffset(gtid.x, gid.x, serialIterations);
                i < finalKeys;
                ++i, t += WaveGetLaneCount() * serialIterations)
        {
    #if defined(PAYLOAD_UINT)
            g_ds[offsets[i]] = b_sortPayload[t];
    #elif defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT)
            g_ds[offsets[i]] = asuint(b_sortPayload[t]);
    #endif
        }
        GroupMemoryBarrierWithGroupSync();
            
        for (uint i = 0, t = SharedOffset(gtid.x, serialIterations);
                i < finalKeys;
                ++i, t += WaveGetLaneCount() * serialIterations)
        {
    #if defined(PAYLOAD_UINT)
            b_altPayload[keys[i]] = g_ds[t];
    #elif defined(PAYLOAD_INT)
            b_altPayload[keys[i]] = asint(g_ds[t]);
    #elif defined(PAYLOAD_FLOAT)
            b_altPayload[keys[i]] = asfloat(g_ds[t]);
    #endif
        }
#else
        const uint lastPartSize = e_numKeys - gid.x * PART_SIZE;
    #if defined(SHOULD_ASCEND)
        for (uint i = gtid.x; i < lastPartSize; i += DS_DIM)
        {
        #if defined(KEY_UINT)
            b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = g_ds[i];
        #elif defined(KEY_INT)
            b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToInt(g_ds[i]);
        #elif defined(KEY_FLOAT)
            b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToFloat(g_ds[i]);
        #endif
        }
    #else
        if (e_radixShift == 24)
        {
            for (uint i = gtid.x; i < lastPartSize; i += DS_DIM)
            {
        #if defined(KEY_UINT)
                b_alt[e_numKeys - g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] - i - 1] = g_ds[i];
        #elif defined(KEY_INT)
                b_alt[e_numKeys - g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] - i - 1] = UintToInt(g_ds[i]);
        #elif defined(KEY_FLOAT)
                b_alt[e_numKeys - g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] - i - 1] = UintToFloat(g_ds[i]);
        #endif
            }
        }
        else
        {
            for (uint i = gtid.x; i < lastPartSize; i += DS_DIM)
            {
        #if defined(KEY_UINT)
                b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = g_ds[i];
        #elif defined(KEY_INT)
                b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToInt(g_ds[i]);
        #elif defined(KEY_FLOAT)
                b_alt[g_ds[ExtractDigit(g_ds[i]) + PART_SIZE] + i] = UintToFloat(g_ds[i]);
        #endif
            }
        }
    #endif
#endif  
    }
}