/******************************************************************************
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/13/2023
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 ******************************************************************************/
//Compiler Defines
//#define KEY_UINT KEY_INT KEY_FLOAT
//#define PAYLOAD_UINT PAYLOAD_INT PAYLOAD_FLOAT
//#define SHOULD_ASCEND
//#define SORT_PAIRS

//General macros 
#define PART_SIZE       3840U   //size of a partition tile

#define US_DIM          128U    //The number of threads in a Upsweep threadblock
#define SCAN_DIM        128U    //The number of threads in a Scan threadblock
#define DS_DIM          256U    //The number of threads in a Downsweep threadblock

#define RADIX           256U    //Number of digit bins
#define RADIX_MASK      255U    //Mask of digit bins
#define RADIX_LOG       8U      //log2(RADIX)

#define HALF_RADIX      128U    //For smaller waves where bit packing is necessary
#define HALF_MASK       127U    // '' 

//For the downsweep kernels
#define DS_KEYS_PER_THREAD  15U     //The number of keys per thread in a Downsweep Threadblock
#define MAX_DS_SMEM         4096U   //shared memory for downsweep kernel

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_radixShift;
    uint e_threadBlocks;
    uint e_seed;
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

inline uint DeviceOffsetWGE16(uint gtid, uint gid)
{
    return SharedOffsetWGE16(gtid) + gid * PART_SIZE;
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

inline uint DeviceOffsetWLT16(uint gtid, uint gid, uint _serialIterations)
{
    return SharedOffsetWLT16(gtid, _serialIterations) + gid * PART_SIZE;
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
    }
    
    //Larger 16 or greater can perform a more elegant scan because 16 * 16 = 256
    if (WaveGetLaneCount() >= 16)
    {
        for (uint i = gtid.x; i < RADIX; i += US_DIM)
            g_us[i] += WavePrefixSum(g_us[i]);
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < (RADIX / WaveGetLaneCount()))
        {
            g_us[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_us[(gtid.x + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        //atomically add to global histogram
        const uint deviceOffset = e_radixShift << 5;
        const uint laneMask = WaveGetLaneCount() - 1;
        const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
        for (uint i = gtid.x; i < RADIX; i += US_DIM)
        {
            const uint index = circularLaneShift + (i & ~laneMask);
            InterlockedAdd(b_globalHist[index + deviceOffset],
                (WaveGetLaneIndex() != laneMask ? g_us[i] : 0) +
                (i >= WaveGetLaneCount() ? WaveReadLaneAt(g_us[i - 1], 0) : 0));
        }
    }
    
    //Exclusive Brent-Kung with fused upsweep downsweep
    if (WaveGetLaneCount() < 16)
    {
        const uint deviceOffset = e_radixShift << 5;
        for (uint i = gtid.x; i < RADIX; i += US_DIM)
            g_us[i] += WavePrefixSum(g_us[i]);
        
        if (gtid.x < WaveGetLaneCount())
        {
            InterlockedAdd(b_globalHist[gtid.x + deviceOffset],
                gtid.x ? g_us[gtid.x - 1] : 0);
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint laneLog = countbits(WaveGetLaneCount() - 1);
        uint offset = laneLog;
        uint j = WaveGetLaneCount();
        for (; j < (RADIX >> 1); j <<= laneLog)
        {
            for (uint i = gtid.x; i < (RADIX >> offset); i += US_DIM)
            {
                g_us[((i + 1) << offset) - 1] +=
                    WavePrefixSum(g_us[((i + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();
            
            for (uint i = gtid.x + j; i < RADIX; i += US_DIM)
            {
                if ((i & ((j << laneLog) - 1)) >= j)
                {
                    if (i < (j << laneLog))
                    {
                        InterlockedAdd(b_globalHist[i + deviceOffset],
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
        
        for (uint i = gtid.x + j; i < RADIX; i += US_DIM)
        {
            InterlockedAdd(b_globalHist[i + deviceOffset],
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
            b_passHist[index + offset] = (WaveGetLaneIndex() != laneMask ? g_scan[gtid.x] : 0) +
                (gtid.x >= WaveGetLaneCount() ? g_scan[(gtid.x & ~laneMask) - 1] : 0) + aggregate;
        }
    }

    if (WaveGetLaneCount() < 16)
    {
        uint aggregate = 0;
        const uint partitions = e_threadBlocks / SCAN_DIM;
        const uint deviceOffset = gid.x * e_threadBlocks;
        const uint laneLog = countbits(WaveGetLaneCount() - 1);
        
        uint k = 0;
        for (; k < partitions; ++k)
        {
            g_scan[gtid.x] = b_passHist[gtid.x + k * SCAN_DIM + deviceOffset];
            g_scan[gtid.x] += WavePrefixSum(g_scan[gtid.x]);
            
            if (gtid.x < WaveGetLaneCount())
            {
                b_passHist[gtid.x + k * SCAN_DIM + deviceOffset] =
                    (gtid.x ? g_scan[gtid.x - 1] : 0) + aggregate;
            }
            GroupMemoryBarrierWithGroupSync();
            
            uint offset = laneLog;
            uint j = WaveGetLaneCount();
            for (; j < (SCAN_DIM >> 1); j <<= laneLog)
            {
                for (uint i = gtid.x; i < (SCAN_DIM >> offset); i += SCAN_DIM)
                {
                    g_scan[((i + 1) << offset) - 1] +=
                        WavePrefixSum(g_scan[((i + 1) << offset) - 1]);
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
            
            if (gtid.x < WaveGetLaneCount())
            {
                b_passHist[gtid.x + k * SCAN_DIM + deviceOffset] =
                    (gtid.x ? g_scan[gtid.x - 1] : 0) + aggregate;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        
        uint offset = laneLog;
        for (uint j = WaveGetLaneCount(); j < finalPartSize; j <<= laneLog)
        {
            for (uint i = gtid.x; i < (finalPartSize >> offset); i += SCAN_DIM)
            {
                g_scan[((i + 1) << offset) - 1] +=
                    WavePrefixSum(g_scan[((i + 1) << offset) - 1]);
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
    if (gid.x < e_threadBlocks - 1)
    {
        uint keys[DS_KEYS_PER_THREAD];
        uint offsets[DS_KEYS_PER_THREAD];
        
        if (WaveGetLaneCount() >= 16)
        {
            //Load keys into registers
            [unroll]
            for (uint i = 0, t = DeviceOffsetWGE16(gtid.x, gid.x);
                 i < DS_KEYS_PER_THREAD;
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
                    for (int wavePart = 0; wavePart < waveParts; ++wavePart)
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
            GroupMemoryBarrierWithGroupSync();
            
            //inclusive/exclusive prefix sum up the histograms
            //followed by exclusive prefix sum across the reductions
            uint reduction = g_ds[gtid.x];
            for (uint i = gtid.x + RADIX; i < WaveHistsSizeWGE16(); i += RADIX)
            {
                reduction += g_ds[i];
                g_ds[i] = reduction - g_ds[i];
            }
            
            reduction += WavePrefixSum(reduction);
            GroupMemoryBarrierWithGroupSync();

            const uint laneMask = WaveGetLaneCount() - 1;
            g_ds[((WaveGetLaneIndex() + 1) & laneMask) + (gtid.x & ~laneMask)] = reduction;
            GroupMemoryBarrierWithGroupSync();
                
            if (gtid.x < RADIX / WaveGetLaneCount())
            {
                g_ds[gtid.x * WaveGetLaneCount()] =
                    WavePrefixSum(g_ds[gtid.x * WaveGetLaneCount()]);
            }
            GroupMemoryBarrierWithGroupSync();
                
            if (WaveGetLaneIndex())
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
            const uint exclusiveWaveReduction = g_ds[gtid.x];
            GroupMemoryBarrierWithGroupSync();
            
            //scatter keys into shared memory
            for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
                g_ds[offsets[i]] = keys[i];
        
            g_ds[gtid.x + PART_SIZE] = b_globalHist[gtid.x + (e_radixShift << 5)] +
                    b_passHist[gtid.x * e_threadBlocks + gid.x] - exclusiveWaveReduction;
            GroupMemoryBarrierWithGroupSync();
            
#if defined(SORT_PAIRS)
            [unroll]
            for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
                 i < DS_KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
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
            GroupMemoryBarrierWithGroupSync();
                
            [unroll]
            for (uint i = 0, t = DeviceOffsetWGE16(gtid.x, gid.x);
                 i < DS_KEYS_PER_THREAD; 
                 ++i, t += WaveGetLaneCount())
            {
    #if defined(PAYLOAD_UINT)
                g_ds[offsets[i]] = b_sortPayload[t];
    #elif defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT)
                g_ds[offsets[i]] = asuint(b_sortPayload[t]);
    #endif
            }
            GroupMemoryBarrierWithGroupSync();
            
            [unroll]
            for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
                 i < DS_KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
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
#endif
        }
        
        if (WaveGetLaneCount() < 16)
        {
            const uint serialIterations = (DS_DIM / WaveGetLaneCount() + 31) / 32;
            
            //Load keys into registers
            [unroll]
            for (uint i = 0, t = DeviceOffsetWLT16(gtid.x, gid.x, serialIterations);
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
                for (int i = gtid.x; i < j; i += DS_DIM)
                {
                    g_ds[((((i << 1) + 2) << shift) - 1) >> 1] +=
                            g_ds[((((i << 1) + 1) << shift) - 1) >> 1] & 0xffff0000;
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
                for (uint i = gtid.x; i < j; i += DS_DIM)
                {
                    const uint t = ((((i << 1) + 1) << shift) - 1) >> 1;
                    const uint t2 = ((((i << 1) + 2) << shift) - 1) >> 1;
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
            
            const uint exclusiveWaveReduction = g_ds[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
            GroupMemoryBarrierWithGroupSync();
            
            //scatter keys into shared memory
            for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
                g_ds[offsets[i]] = keys[i];
        
            g_ds[gtid.x + PART_SIZE] = b_globalHist[gtid.x + (e_radixShift << 5)] +
                    b_passHist[gtid.x * e_threadBlocks + gid.x] - exclusiveWaveReduction;
            GroupMemoryBarrierWithGroupSync();
        
            //scatter runs of keys into device memory, 
            //store the scatter location in the key register to reuse for the payload
#if defined(SORT_PAIRS)
            [unroll]
            for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
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
            GroupMemoryBarrierWithGroupSync();
                
            [unroll]
            for (uint i = 0, t = DeviceOffsetWLT16(gtid.x, gid.x, serialIterations);
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
            for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
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
#endif
        }
    }
    
    //perform the sort on the final partition slightly differently 
    //to handle input sizes not perfect multiples of the partition
    if (gid.x == e_threadBlocks - 1)
    {
        //load the global and pass histogram values into shared memory
        if (gtid.x < RADIX)
        {
            g_ds[gtid.x] = b_globalHist[gtid.x + (e_radixShift << 5)] +
                b_passHist[gtid.x * e_threadBlocks + gid.x];
        }
        GroupMemoryBarrierWithGroupSync();
        
        const uint waveParts = (WaveGetLaneCount() + 31) / 32;
        for (int i = gtid.x + gid.x * PART_SIZE; i < (gid.x + 1) * PART_SIZE; i += DS_DIM)
        {
            uint key;
            if (i < e_numKeys)
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
            uint offset;
            uint bits = 0;
            if (i < e_numKeys)
            {
                [unroll]
                for (uint k = 0; k < RADIX_LOG; ++k)
                {
                    const bool t = key >> (k + e_radixShift) & 1;
                    const uint4 ballot = WaveActiveBallot(t);
                    for (int wavePart = 0; wavePart < waveParts; ++wavePart)
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
            
            for (int k = 0; k < DS_DIM / WaveGetLaneCount(); ++k)
            {
                if (getWaveIndex(gtid.x) == k && i < e_numKeys)
                    offset = g_ds[key >> e_radixShift & RADIX_MASK] + bits;
                GroupMemoryBarrierWithGroupSync();
                
                if (getWaveIndex(gtid.x) == k && i < e_numKeys && bits == 0)
                {
                    for (int wavePart = 0; wavePart < waveParts; ++wavePart)
                        g_ds[key >> e_radixShift & RADIX_MASK] += countbits(waveFlags[wavePart]);
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (i < e_numKeys)
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
        }
    }
}

//---------------------------VALIDATION UTILITIES---------------------------
//Perform validation on GPU to massively increase test speed
//Note: Replaces the e_seed value with maxReadbackErrors, so
//we do not need a larger constant buffer bound !
RWStructuredBuffer<uint> b_errorCount : register(u6);
RWStructuredBuffer<uint3> b_error : register(u7);

#define VAL_PART_SIZE   2048
#define VAL_THREADS     256

#define TAUS_STEP_1 ((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19)
#define TAUS_STEP_2 ((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25)
#define TAUS_STEP_3 ((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11)
#define LCG_STEP    (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS (z1 ^ z2 ^ z3 ^ z4)

#if defined(KEY_UINT)
groupshared uint g_val[VAL_PART_SIZE + 1];
#elif defined(KEY_INT)
groupshared int g_val[VAL_PART_SIZE + 1];
#elif defined(KEY_FLOAT)
groupshared float g_val[VAL_PART_SIZE + 1];
#endif

//Initialize the input on the GPU, assumes threadblocks = 256
[numthreads(VAL_THREADS, 1, 1)]
void InitSortInput(int3 id : SV_DispatchThreadID)
{
    const uint numKeys = e_numKeys;
    const uint inc = VAL_THREADS * 256;
    
    uint z1 = (id.x << 2) * e_seed;
    uint z2 = ((id.x << 2) + 1) * e_seed;
    uint z3 = ((id.x << 2) + 2) * e_seed;
    uint z4 = ((id.x << 2) + 3) * e_seed;
    
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;
    
    for (uint i = id.x; i < numKeys; i += inc)
    {
        z1 = TAUS_STEP_1;
        z2 = TAUS_STEP_2;
        z3 = TAUS_STEP_3;
        z4 = LCG_STEP;
#if defined(KEY_UINT)
        b_sort[i] = HYBRID_TAUS;
#elif defined (KEY_INT)
        b_sort[i] = asint(HYBRID_TAUS);
#elif defined (KEY_FLOAT)
        b_sort[i] = asfloat(HYBRID_TAUS);
#endif
        
#if defined(SORT_PAIRS)
    #if defined(PAYLOAD_UINT)
        b_sortPayload[i] = HYBRID_TAUS;
    #elif defined (PAYLOAD_INT)
        b_sortPayload[i] = asint(HYBRID_TAUS);
    #elif defined (PAYLOAD_FLOAT)
        b_sortPayload[i] = asfloat(HYBRID_TAUS);
    #endif
#endif
    }
}

//Used to standalone test the scan kernel, assumes threadblocks = 1
//Scan values so small its not a huge time sink to check on the CPU
[numthreads(VAL_THREADS, 1, 1)]
void InitScanTestValues(int3 id : SV_DispatchThreadID)
{
    if (id.x < e_numKeys)
        b_passHist[id.x] = 1;
}

[numthreads(1, 1, 1)]
void ClearErrorCount(int3 id : SV_DispatchThreadID)
{
    b_errorCount[0] = 0;
}

[numthreads(VAL_THREADS, 1, 1)]
void Validate(int3 gtid : SV_GroupThreadID, int3 gid : SV_GroupID)
{
    if (gid.x < e_threadBlocks - 1)
    {
        const uint t = gid.x * VAL_PART_SIZE;
        for (int i = gtid.x; i < VAL_PART_SIZE + 1; i += VAL_THREADS)
        {
#if defined(KEY_UINT)
            g_val[i] = b_sort[i + t];
#elif defined (KEY_INT)
            g_val[i] = b_sort[i + t];
#elif defined (KEY_FLOAT)
            g_val[i] = b_sort[i + t];
#endif
        }
        GroupMemoryBarrierWithGroupSync();
        
        for (int i = gtid.x; i < VAL_PART_SIZE; i += VAL_THREADS)
        {
#if defined(SHOULD_ASCEND)
    #if defined(KEY_UINT) || defined(KEY_INT) || defined(KEY_FLOAT)
            if (g_val[i] > g_val[i + 1])
    #endif
#else
    #if defined(KEY_UINT) || defined(KEY_INT) || defined(KEY_FLOAT)
            if (g_val[i] < g_val[i + 1])
    #endif
#endif
            {
                InterlockedAdd(b_errorCount[0], 1);
            }
        }
    }
    else
    {
        for (int i = gtid.x + gid.x * VAL_PART_SIZE; i < e_numKeys - 1; i += VAL_THREADS)
        {
            
#if defined(SHOULD_ASCEND)
    #if defined(KEY_UINT) || defined(KEY_INT) || defined(KEY_FLOAT)
            if (b_sort[i] > b_sort[i + 1])
    #endif
#else
    #if defined(KEY_UINT) || defined(KEY_INT) || defined(KEY_FLOAT)
            if (b_sort[i] < b_sort[i + 1])
    #endif
#endif
            {
                InterlockedAdd(b_errorCount[0], 1);
            }
        }
    }
}