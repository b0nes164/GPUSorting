/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/28/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
//Compiler Defines
//#define KEY_UINT KEY_INT KEY_FLOAT
//#define PAYLOAD_UINT PAYLOAD_INT PAYLOAD_FLOAT
//#define SHOULD_ASCEND
//#define SORT_PAIRS
//#define ENABLE_16_BIT
#include "SortCommon.hlsl"

#pragma kernel InitSweep
#pragma kernel GlobalHistogram
#pragma kernel Scan

#define G_HIST_PART_SIZE    32768U  //The size of a GlobalHistogram partition tile.
#define G_HIST_DIM          128U    //The number of threads in a global hist threadblock

#define SEC_RADIX_START     256     //Offset for retrieving value from global histogram buffer
#define THIRD_RADIX_START   512     //Offset for retrieving value from global histogram buffer
#define FOURTH_RADIX_START  768     //Offset for retrieving value from global histogram buffer

#define FLAG_NOT_READY      0       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3       //Mask used to retrieve flag values

#define MAX_SPIN_COUNT      1       //How much should we allow threadblocks to wait on preceeding tiles?

RWStructuredBuffer<uint> b_globalHist;                  //buffer holding device level offsets for each binning pass
globallycoherent RWStructuredBuffer<uint> b_passHist;   //buffer used to store reduced sums of partition tiles
globallycoherent RWStructuredBuffer<uint> b_index;      //buffer used to atomically assign partition tile indexes

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
void InitSweep(uint3 id : SV_DispatchThreadID)
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
        e_numKeys : (gid + 1) * G_HIST_PART_SIZE;
    
    uint t;
    for (uint i = gtid + gid * G_HIST_PART_SIZE; i < partitionEnd; i += G_HIST_DIM)
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
        
    //If RADIX is not a power of lanecount
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
inline void AssignPartitionTile(uint gtid, inout uint partitionIndex)
{
    if (!gtid)
        InterlockedAdd(b_index[CurrentPass()], 1, g_d[D_TOTAL_SMEM - 1]);
    GroupMemoryBarrierWithGroupSync();
    partitionIndex = g_d[D_TOTAL_SMEM - 1];
}

//For OneSweep
inline void DeviceBroadcastReductionsWGE16(uint gtid, uint partIndex, uint histReduction)
{
    if (partIndex < e_threadBlocks - 1)
    {
        InterlockedAdd(b_passHist[gtid + PassHistOffset(partIndex + 1)],
            FLAG_REDUCTION | histReduction << 2);
    }
}

inline void DeviceBroadcastReductionsWLT16(uint gtid, uint partIndex, uint histReduction)
{
    if (partIndex < e_threadBlocks - 1)
    {
        InterlockedAdd(b_passHist[(gtid << 1) + PassHistOffset(partIndex + 1)],
            FLAG_REDUCTION | (histReduction & 0xffff) << 2);
                
        InterlockedAdd(b_passHist[(gtid << 1) + 1 + PassHistOffset(partIndex + 1)],
            FLAG_REDUCTION | (histReduction >> 16 & 0xffff) << 2);
    }
}

//For ForwardSweep
inline void CASDeviceBroadcastReductionsWGE16(uint gtid, uint partIndex, uint histReduction)
{
    if (partIndex < e_threadBlocks - 1)
    {
        InterlockedCompareStore(b_passHist[gtid + PassHistOffset(partIndex + 1)], 0,
            FLAG_REDUCTION | histReduction << 2);
    }
}

inline void CASDeviceBroadcastReductionsWLT16(uint gtid, uint partIndex, uint histReduction)
{
    if (partIndex < e_threadBlocks - 1)
    {
        InterlockedCompareStore(b_passHist[gtid + PassHistOffset(partIndex + 1)], 0,
            FLAG_REDUCTION | (histReduction & 0xffff) << 2);
        
        InterlockedCompareStore(b_passHist[gtid + PassHistOffset(partIndex + 1)], 0,
            FLAG_REDUCTION | (histReduction >> 16 & 0xffff) << 2);
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
                g_d[gtid + PART_SIZE] = lookbackReduction - exclusiveHistReduction;
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

inline void InitializeLookbackFallback(uint gtid)
{
    if (gtid == 0)
        g_d[0] = 0;
    
    if (gtid == 1)
        g_d[1] = 0;
}

inline void FallbackHistogram(uint index)
{
#if defined(KEY_UINT)
    InterlockedAdd(g_d[ExtractDigit(b_sort[index], e_radixShift) + RADIX], 1);
#elif defined(KEY_INT)
    InterlockedAdd(g_d[ExtractDigit(IntToUint(b_sort[index]), e_radixShift) + RADIX], 1);
#elif defined(KEY_FLOAT)
    InterlockedAdd(g_d[ExtractDigit(FloatToUint(b_sort[index]), e_radixShift) + RADIX], 1);
#endif
}

inline void LookbackWithFallback(uint gtid, uint partIndex, uint exclusiveHistReduction)
{
    uint spinCount = 0;
    uint lookbackReduction = 0;
    bool lookbackComplete = gtid < RADIX ? false : true;
    bool warpLookbackComplete = gtid < RADIX ? false : true;
    uint lookbackIndex = (gtid & RADIX_MASK) + PassHistOffset(partIndex);
    
    while (g_d[0] < RADIX / WaveGetLaneCount())
    {
        //Try to read the preceeding tiles
        uint flagPayload;
        if (!warpLookbackComplete)
        {
            if (!lookbackComplete)
            {
                while (spinCount < MAX_SPIN_COUNT)
                {
                    flagPayload = b_passHist[lookbackIndex];
                    if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
                        break;
                    else
                        spinCount++;
                }
            }
            
            //Did we encounter any deadlocks?
            if (WaveActiveAnyTrue(spinCount == MAX_SPIN_COUNT) && !WaveGetLaneIndex())
                InterlockedOr(g_d[1], 1);
        }
        GroupMemoryBarrierWithGroupSync();
        
        //Yes: fallback
        if (g_d[1])
        {
            if (gtid < RADIX)
                g_d[gtid + RADIX] = 0;
            GroupMemoryBarrierWithGroupSync();
            if (!gtid)
                g_d[1] = 0;
            
            const uint fallbackEnd = PART_SIZE * ((lookbackIndex >> RADIX_LOG) - e_threadBlocks * CurrentPass());
            for (uint i = gtid + PART_SIZE * ((lookbackIndex >> RADIX_LOG) - e_threadBlocks * CurrentPass() - 1); i < fallbackEnd; i += D_DIM)
                FallbackHistogram(i);
            GroupMemoryBarrierWithGroupSync();
            
            uint reduceOut;
            if (gtid < RADIX)
            {
                InterlockedCompareExchange(b_passHist[gtid + PassHistOffset((lookbackIndex >> RADIX_LOG) - e_threadBlocks * CurrentPass())], 0,
                    FLAG_REDUCTION | g_d[gtid + RADIX] << 2, reduceOut);
            }
            
            if (!lookbackComplete)
            {
                if ((reduceOut & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    lookbackReduction += reduceOut >> 2;
                    if (partIndex < e_threadBlocks - 1)
                    {
                        InterlockedAdd(b_passHist[gtid + PassHistOffset(partIndex + 1)],
                            1 | lookbackReduction << 2);
                    }
                    lookbackComplete = true;
                }
                else
                {
                    lookbackReduction += g_d[gtid + RADIX];
                }
            }
            
            spinCount = 0;
        }
        else //No: proceed as normal
        {
            if (!lookbackComplete)
            {
                lookbackReduction += flagPayload >> 2;
                if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    if (partIndex < e_threadBlocks - 1)
                    {
                        InterlockedAdd(b_passHist[gtid + PassHistOffset(partIndex + 1)],
                            1 | lookbackReduction << 2);
                    }
                    lookbackComplete = true;
                }
                else
                {
                    spinCount = 0;
                }
            }
        }
        lookbackIndex -= RADIX; //The threadblock lookbacks in lockstep.
        
        //Have all digits completed their lookbacks?
        if (!warpLookbackComplete)
        {
            warpLookbackComplete = WaveActiveAllTrue(lookbackComplete);
            if (warpLookbackComplete && !WaveGetLaneIndex())
                InterlockedAdd(g_d[0], 1);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    //Post results into shared memory
    if (gtid < RADIX)
        g_d[gtid + PART_SIZE] = lookbackReduction - exclusiveHistReduction;
}