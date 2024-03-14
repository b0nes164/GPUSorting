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
#include "SortCommon.hlsl"

#define G_HIST_DIM          128U    //The number of threads in a global hist threadblock

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3       //Mask used to retrieve flag values

RWStructuredBuffer<uint> b_globalHist                   : register(u4); //buffer holding device level offsets for each binning pass
globallycoherent RWStructuredBuffer<uint> b_passHist    : register(u5); //buffer used to store reduced sums of partition tiles
globallycoherent RWStructuredBuffer<uint> b_index       : register(u6); //buffer used to atomically assign partition tile indexes

groupshared uint4 g_gHist[RADIX * 2];   //Shared memory for GlobalHistogram
groupshared uint g_scan[RADIX];         //Shared memory for Scan  


inline uint CurrentPass()
{
    return e_radixShift >> 3;
}

inline uint PassHistOffset(uint index)
{
    return ((CurrentPass() * e_threadBlocks) + index) << RADIX_LOG;
}

[numthreads(256, 1, 1)]
void InitOneSweep(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    const uint clearEnd = e_threadBlocks * RADIX * RADIX_PASSES;
    for (uint i = id.x; i < clearEnd; i += increment)
        b_passHist[i] = 0;

    if (id.x < RADIX * RADIX_PASSES)
        b_globalHist[id.x] = 0;
    
    if (id.x < RADIX_PASSES)
        b_index[id.x] = 0;
}

//*****************************************************************************
//GLOBAL HISTOGRAM KERNEL
//*****************************************************************************
//histogram, 64 threads to a histogram
inline void HistogramDigitCounts(uint gtid, uint gid)
{
    const uint histOffset = gtid / 64 * RADIX;
    const uint partitionEnd = gid == e_threadBlocks - 1 ?
        e_numKeys : (gid + 1) * PART_SIZE;
    
    uint t;
    for (uint i = gtid + gid * PART_SIZE; i < partitionEnd; i += G_HIST_DIM)
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
}

//reduce counts and atomically add to device
inline void ReduceWriteDigitCounts(uint gtid)
{
    for (uint i = gtid; i < RADIX; i += G_HIST_DIM)
    {
        InterlockedAdd(b_globalHist[i], g_gHist[i].x + g_gHist[i + RADIX].x);
        InterlockedAdd(b_globalHist[i + SEC_RADIX_START], g_gHist[i].y + g_gHist[i + RADIX].y);
        InterlockedAdd(b_globalHist[i + THIRD_RADIX_START], g_gHist[i].z + g_gHist[i + RADIX].z);
        InterlockedAdd(b_globalHist[i + FOURTH_RADIX_START], g_gHist[i].w + g_gHist[i + RADIX].w);
    }
}

[numthreads(G_HIST_DIM, 1, 1)]
void GlobalHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    //clear shared memory
    const uint histsEnd = RADIX * 2;
    for (uint i = gtid.x; i < histsEnd; i += G_HIST_DIM)
        g_gHist[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    
    HistogramDigitCounts(gtid.x, gid.x);
    GroupMemoryBarrierWithGroupSync();
    
    ReduceWriteDigitCounts(gtid.x);
}

//*****************************************************************************
//SCAN KERNEL
//*****************************************************************************
inline void LoadInclusiveScan(uint gtid, uint gid)
{
    const uint t = b_globalHist[gtid + gid * RADIX];
    g_scan[gtid] = t + WavePrefixSum(t);
}

inline void GlobalHistExclusiveScanWGE16(uint gtid, uint gid)
{
    GroupMemoryBarrierWithGroupSync();
    if (gtid < (RADIX / WaveGetLaneCount()))
    {
        g_scan[(gtid + 1) * WaveGetLaneCount() - 1] +=
            WavePrefixSum(g_scan[(gtid + 1) * WaveGetLaneCount() - 1]);
    }
    GroupMemoryBarrierWithGroupSync();
        
    const uint laneMask = WaveGetLaneCount() - 1;
    const uint index = (WaveGetLaneIndex() + 1 & laneMask) + (gtid & ~laneMask);
    b_passHist[index + gid * RADIX * e_threadBlocks] =
        ((WaveGetLaneIndex() != laneMask ? g_scan[gtid] : 0) +
        (gtid >= WaveGetLaneCount() ? WaveReadLaneAt(g_scan[gtid - 1], 0) : 0)) << 2 | FLAG_INCLUSIVE;
}

inline void GlobalHistExclusiveScanWLT16(uint gtid, uint gid)
{
    const uint passHistOffset = gid * RADIX * e_threadBlocks;
    if (gtid < WaveGetLaneCount())
    {
        const uint circularLaneShift = WaveGetLaneIndex() + 1 &
            WaveGetLaneCount() - 1;
        b_passHist[circularLaneShift + passHistOffset] =
            (circularLaneShift ? g_scan[gtid] : 0) << 2 | FLAG_INCLUSIVE;
    }
    GroupMemoryBarrierWithGroupSync();
        
    const uint laneLog = countbits(WaveGetLaneCount() - 1);
    uint offset = laneLog;
    uint j = WaveGetLaneCount();
    for (; j < (RADIX >> 1); j <<= laneLog)
    {
        if (gtid < (RADIX >> offset))
        {
            g_scan[((gtid + 1) << offset) - 1] +=
                WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
            
        if ((gtid & ((j << laneLog) - 1)) >= j)
        {
            if (gtid < (j << laneLog))
            {
                b_passHist[gtid + passHistOffset] =
                    (WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) +
                    ((gtid & (j - 1)) ? g_scan[gtid - 1] : 0)) << 2 | FLAG_INCLUSIVE;
            }
            else
            {
                if ((gtid + 1) & (j - 1))
                {
                    g_scan[gtid] +=
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                }
            }
        }
        offset += laneLog;
    }
    GroupMemoryBarrierWithGroupSync();
        
    //If RADIX is not a multiple of lanecount
    const uint index = gtid.x + j;
    if (index < RADIX)
    {
        b_passHist[index + passHistOffset] =
            (WaveReadLaneAt(g_scan[((index >> offset) << offset) - 1], 0) +
            ((index & (j - 1)) ? g_scan[index - 1] : 0)) << 2 | FLAG_INCLUSIVE;
    }
}

[numthreads(RADIX, 1, 1)]
void Scan(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    LoadInclusiveScan(gtid.x, gid.x);
    
    if (WaveGetLaneCount() >= 16)
        GlobalHistExclusiveScanWGE16(gtid.x, gid.x);
    
    if (WaveGetLaneCount() < 16)
        GlobalHistExclusiveScanWLT16(gtid.x, gid.x);
}

//*****************************************************************************
//DIGIT BINNING PASS KERNEL
//*****************************************************************************
inline void ClearWaveHists(uint gtid)
{
    const uint histsEnd = WaveGetLaneCount() >= 16 ?
        WaveHistsSizeWGE16() : WaveHistsSizeWLT16();
    for (uint i = gtid; i < histsEnd; i += PASS_DIM)
        g_pass[i] = 0;
}

inline void AssignPartitionTile(uint gtid, inout uint partitionIndex)
{
    if (!gtid)
        InterlockedAdd(b_index[CurrentPass()], 1, g_pass[PART_SIZE - 1]);
    GroupMemoryBarrierWithGroupSync();
    partitionIndex = g_pass[PART_SIZE - 1];
}

inline void DeviceBroadcastReductionsWGE16(uint gtid, uint partIndex, uint histReduction)
{
    if (partIndex < e_threadBlocks - 1)
    {
        InterlockedAdd(b_passHist[gtid + PassHistOffset(partIndex + 1)],
            FLAG_REDUCTION | histReduction << 2);
    }
}

inline void Lookback(uint gtid, uint partIndex, uint exclusiveHistReduction)
{
    if (gtid < RADIX)
    {
        uint lookbackReduction = 0;
        for (uint k = partIndex; k >= 0;)
        {
            const uint flagPayload = b_passHist[gtid + PassHistOffset(k)];
            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
            {
                lookbackReduction += flagPayload >> 2;
                if (partIndex < e_threadBlocks - 1)
                {
                    InterlockedAdd(b_passHist[gtid + PassHistOffset(partIndex + 1)],
                        1 | lookbackReduction << 2);
                }
                g_pass[gtid + PART_SIZE] = lookbackReduction - exclusiveHistReduction;
                break;
            }
                    
            if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
            {
                lookbackReduction += flagPayload >> 2;
                k--;
            }
        }
    }
}

[numthreads(PASS_DIM, 1, 1)]
void DigitBinningPass(uint3 gtid : SV_GroupThreadID)
{
    const uint serialIterations = SerialIterations();
    uint partitionIndex;
    KeyStruct keys;
    OffsetStruct offsets;
    
    //WGT 16 does not require additional barriers
    if (WaveGetLaneCount() > 16)
    {
        ClearWaveHists(gtid.x);
        AssignPartitionTile(gtid.x, partitionIndex);
    }
    
    if (WaveGetLaneCount() <= 16)
    {
        AssignPartitionTile(gtid.x, partitionIndex);
        GroupMemoryBarrierWithGroupSync();
        ClearWaveHists(gtid.x);
        GroupMemoryBarrierWithGroupSync();
    }
    
    if(partitionIndex < e_threadBlocks - 1)
    {
        if (WaveGetLaneCount() >= 16)
            keys = LoadKeysWGE16(gtid.x, partitionIndex);
        
        if (WaveGetLaneCount() < 16)
            keys = LoadKeysWLT16(gtid.x, partitionIndex, serialIterations);
    }
        
    if(partitionIndex == e_threadBlocks - 1)
    {
        if (WaveGetLaneCount() >= 16)
            keys = LoadKeysPartialWGE16(gtid.x, partitionIndex);
        
        if (WaveGetLaneCount() < 16)
            keys = LoadKeysPartialWLT16(gtid.x, partitionIndex, serialIterations);
    }
    
    uint exclusiveHistReduction;
    if (WaveGetLaneCount() >= 16)
    {
        offsets = RankKeysWGE16(gtid.x, keys);
        GroupMemoryBarrierWithGroupSync();
        
        uint histReduction;
        if (gtid.x < RADIX)
        {
            histReduction = WaveHistInclusiveScanCircularShiftWGE16(gtid.x);
            DeviceBroadcastReductionsWGE16(gtid.x, partitionIndex, histReduction);
            histReduction += WavePrefixSum(histReduction); //take advantage of barrier to begin scan
        }
        GroupMemoryBarrierWithGroupSync();

        WaveHistReductionExclusiveScanWGE16(gtid.x, histReduction);
        GroupMemoryBarrierWithGroupSync();
            
        UpdateOffsetsWGE16(gtid.x, offsets, keys);
        if (gtid.x < RADIX)
            exclusiveHistReduction = g_pass[gtid.x]; //take advantage of barrier to grab value
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (WaveGetLaneCount() < 16)
    {
        //stub
        offsets = RankKeysWGE16(gtid.x, keys);
        GroupMemoryBarrierWithGroupSync();
            
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
                
            if(partitionIndex < e_threadBlocks - 1)
            {
                InterlockedAdd(b_passHist[(gtid.x << 1) + PassHistOffset(partitionIndex + 1)],
                    FLAG_REDUCTION | (histReduction & 0xffff) << 2);
                
                InterlockedAdd(b_passHist[(gtid.x << 1) + 1 + PassHistOffset(partitionIndex + 1)],
                    FLAG_REDUCTION | (histReduction >> 16 & 0xffff) << 2);
            }
        }
            
        uint shift = 1;
        for (uint j = RADIX >> 2; j > 0; j >>= 1)
        {
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < j)
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
            if (gtid.x < j)
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
                const uint t2 = ExtractPackedIndex(keys.k[i]);
                offsets.o[i] += ExtractPackedValue(g_pass[t2 + t] + g_pass[t2], keys.k[i]);
            }
        }
        else
        {
            [unroll]
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                offsets.o[i] += ExtractPackedValue(g_pass[ExtractPackedIndex(keys.k[i])], keys.k[i]);
        }
            
        if (gtid.x < RADIX)
            exclusiveHistReduction = g_pass[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
        GroupMemoryBarrierWithGroupSync();
    }
    
    ScatterKeysShared(offsets, keys);
    Lookback(gtid.x, partitionIndex, exclusiveHistReduction);
    GroupMemoryBarrierWithGroupSync();
    
    //Scatter keys into device
    if(partitionIndex < e_threadBlocks - 1)
    {
        if (WaveGetLaneCount() >= 16)
            ScatterDeviceWGE16(gtid.x, partitionIndex, keys, offsets);
        
        if (WaveGetLaneCount() < 16)
            ScatterDeviceWLT16(gtid.x, partitionIndex, serialIterations, keys, offsets);
    }
        
    if(partitionIndex == e_threadBlocks - 1)
    {
        if (WaveGetLaneCount() >= 16)
            ScatterDevicePartialWGE16(gtid.x, partitionIndex, keys, offsets);
        
        if (WaveGetLaneCount() < 16)
            ScatterDevicePartialWLT16(gtid.x, partitionIndex, serialIterations, keys, offsets);
    }
}