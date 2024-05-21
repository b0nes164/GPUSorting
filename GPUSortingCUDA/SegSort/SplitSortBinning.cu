/******************************************************************************
*  GPUSorting
 * SplitSort
 * Experimental SegSort that does not use cooperative groups
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 5/16/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#include "SplitSortBinning.cuh"

#define NEXT_FIT_BINNING_SIZE   2048    //Number of segments that a partition processes
#define NEXT_FIT_BLOCK_DIM      64      //Block dim of a partition
#define NEXT_FIT_WARPS          2       //Warps in a block
#define NEXT_FIT_SPT            32      //Segments processed per thread
#define MIN_BIN_SIZE            32      //Segments below this size are packed into bins
#define SEG_HIST_SIZE           9       //How many different segment groupings we have

//For the scan
#define FLAG_NOT_READY      0
#define FLAG_REDUCTION      1
#define FLAG_INCLUSIVE      2
#define FLAG_MASK           3

__device__ __forceinline__ void LoadFull(
    const uint32_t* segments,
    uint32_t* threadSegments,
    const uint32_t partIndex)
{
    #pragma unroll
    for (uint32_t i = threadIdx.x * NEXT_FIT_SPT + NEXT_FIT_BINNING_SIZE * partIndex, k = 0;
        k < NEXT_FIT_SPT;
        ++i, ++k)
    {
        threadSegments[k] = segments[i + 1] - segments[i];
    }
}

__device__ __forceinline__ void LoadPartial(
    const uint32_t* segments,
    uint32_t* threadSegments,
    const uint32_t partIndex,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    const uint32_t partEnd = partIndex == gridDim.x - 1 ? totalSegCount : (partIndex + 1) * NEXT_FIT_BINNING_SIZE;
    #pragma unroll
    for (uint32_t i = threadIdx.x * NEXT_FIT_SPT + NEXT_FIT_BINNING_SIZE * partIndex, k = 0;
        k < NEXT_FIT_SPT;
        ++i, ++k)
    {
        if (i < partEnd)
        {
            const uint32_t upper = i == partEnd - 1 ? totalSegLength : segments[i + 1];
            threadSegments[k] = upper - segments[i];
        }
    }
}

__device__ __forceinline__ void LoadSegments(
    const uint32_t* segments,
    uint32_t* threadSegments,
    const uint32_t partIndex,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    //slight wierdness due to exclusive prefix summed segments
    if (gridDim.x > 1)
    {
        if (partIndex < gridDim.x - 1)
            LoadFull(segments, threadSegments, partIndex);
        else
            LoadPartial(segments, threadSegments, partIndex, totalSegCount, totalSegLength);
    }
    else
    {
        LoadPartial(segments, threadSegments, partIndex, totalSegCount, totalSegLength);
    }
}

//Simplest approach, which tries to preserve the locality
//of the segments to optimize memory accesses.
__device__ __forceinline__ void PackBin(
    uint32_t* threadSegments,
    uint32_t* s_warpHist,
    uint32_t& currentBinTotal,
    uint32_t& currentBinCount,
    uint32_t& minBinsPacked,
    const uint32_t i)
{
    //if the segment is large enough, immediately start a new bin
    if (threadSegments[i] > MIN_BIN_SIZE)
    {
        //Enter the large bin, using BBSegSort style binning
        if (threadSegments[i] <= 64)
            atomicAdd((uint32_t*)&s_warpHist[1], 1);
        if (64 < threadSegments[i] && threadSegments[i] <= 128)
            atomicAdd((uint32_t*)&s_warpHist[2], 1);
        if (128 < threadSegments[i] && threadSegments[i] <= 256)
            atomicAdd((uint32_t*)&s_warpHist[3], 1);
        if (256 < threadSegments[i] && threadSegments[i] <= 512)
            atomicAdd((uint32_t*)&s_warpHist[4], 1);
        if (512 < threadSegments[i] && threadSegments[i] <= 1024)
            atomicAdd((uint32_t*)&s_warpHist[5], 1);
        if (1024 < threadSegments[i] && threadSegments[i] <= 2048)
            atomicAdd((uint32_t*)&s_warpHist[6], 1);
        if (2048 < threadSegments[i] && threadSegments[i] <= 4096)
            atomicAdd((uint32_t*)&s_warpHist[7], 1);
        if (4096 < threadSegments[i])
            atomicAdd((uint32_t*)&s_warpHist[8], 1);

        //End the current bin
        if (currentBinCount)
        {
            reinterpret_cast<uint16_t*>(threadSegments)[minBinsPacked * 2] = currentBinCount;
            reinterpret_cast<uint16_t*>(threadSegments)[minBinsPacked * 2 + 1] = i - currentBinCount;
            atomicAdd((uint32_t*)&s_warpHist[0], 1);

            minBinsPacked++;
            currentBinCount = 0;
            currentBinTotal = 0;
        }
    }
    else
    {
        //Does this fit in the bin?
        if (threadSegments[i] + currentBinTotal <= MIN_BIN_SIZE)
        {
            //Yes
            currentBinTotal += threadSegments[i];
            currentBinCount++;
        }
        else
        {
            //No, end the current bin, start a new bin
            reinterpret_cast<uint16_t*>(threadSegments)[minBinsPacked * 2] = currentBinCount;
            reinterpret_cast<uint16_t*>(threadSegments)[minBinsPacked * 2 + 1] = i - currentBinCount;
            atomicAdd((uint32_t*)&s_warpHist[0], 1);

            minBinsPacked++;
            currentBinCount = 1;
            currentBinTotal = threadSegments[i];
        }
    }
}

//Is there a "hanging" bin leftover?
__device__ __forceinline__ void AttemptPackHangingBin(
    uint32_t* threadSegments,
    uint32_t* s_warpHist,
    uint32_t& minBinsPacked,
    const uint32_t currentBinCount,
    const uint32_t threadRunLength)
{
    if (currentBinCount)
    {
        reinterpret_cast<uint16_t*>(threadSegments)[minBinsPacked * 2] = currentBinCount;
        reinterpret_cast<uint16_t*>(threadSegments)[minBinsPacked * 2 + 1] = threadRunLength - currentBinCount;
        atomicAdd((uint32_t*)&s_warpHist[0], 1);

        minBinsPacked++;
    }
}

__device__ __forceinline__ void NextFitBinPackFull(
    uint32_t* threadSegments,
    uint32_t* s_warpHist,
    uint32_t& minBinsPacked)
{
    uint32_t currentBinTotal = 0;   //What is the total length of segments in this current bin
    uint32_t currentBinCount = 0;   //How many segments are in this current bin?
    
    #pragma unroll
    for (uint32_t i = 0; i < NEXT_FIT_SPT; ++i)
    {
        PackBin(
            threadSegments,
            s_warpHist,
            currentBinTotal,
            currentBinCount,
            minBinsPacked,
            i);
    }

    AttemptPackHangingBin(
        threadSegments,
        s_warpHist,
        minBinsPacked,
        currentBinCount,
        NEXT_FIT_SPT);
}

__device__ __forceinline__ void NextFitBinPackPartial(
    uint32_t* threadSegments,
    uint32_t* s_warpHist,
    uint32_t& minBinsPacked,
    const uint32_t partitionSize)
{
    uint32_t currentBinTotal = 0;   //What is the total length of segments in this current bin
    uint32_t currentBinCount = 0;   //How many segments are in this current bin?

    int32_t threadRunLength = partitionSize - threadIdx.x * NEXT_FIT_SPT;
    if (threadRunLength > NEXT_FIT_SPT)
        threadRunLength = NEXT_FIT_SPT;
    if (threadRunLength < 0)
        threadRunLength = 0;

    for (uint32_t i = 0; i < (uint32_t)threadRunLength; ++i)
    {
        PackBin(
            threadSegments,
            s_warpHist,
            currentBinTotal,
            currentBinCount,
            minBinsPacked,
            i);
    }

    AttemptPackHangingBin(
        threadSegments,
        s_warpHist,
        minBinsPacked,
        currentBinCount,
        (uint32_t)threadRunLength);
}

__device__ __forceinline__ void NextFitBinPack(
    uint32_t* threadSegments,
    uint32_t* s_warpHist,
    uint32_t& minBinsPacked,
    const uint32_t partIndex,
    const uint32_t totalSegCount)
{
    if (partIndex < gridDim.x - 1)
        NextFitBinPackFull(threadSegments, s_warpHist, minBinsPacked);
    else
        NextFitBinPackPartial(threadSegments, s_warpHist, minBinsPacked, totalSegCount - NEXT_FIT_BINNING_SIZE * partIndex);
}

//to do: move histogramming from shared to reg
__global__ void SplitSortBinning::NextFitBinPacking(
    const uint32_t* segments,
    uint32_t* segHist,
    uint32_t* minBinSegCounts,
    uint32_t* binOffsets,
    volatile uint32_t* index,
    volatile uint32_t* reduction,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    __shared__ uint32_t s_hist[SEG_HIST_SIZE * NEXT_FIT_WARPS];
    __shared__ uint32_t s_reduction[NEXT_FIT_BLOCK_DIM];
    __shared__ uint32_t s_broadcast;

    uint32_t* s_warpHist = &s_hist[SEG_HIST_SIZE * WARP_INDEX];
    if (getLaneId() < SEG_HIST_SIZE)
        s_warpHist[getLaneId()] = 0;

    //do the chained scan thing
    if (!threadIdx.x)
        s_broadcast = atomicAdd((uint32_t*)&index[0], 1);
    __syncthreads();
    const uint32_t partitionIndex = s_broadcast;

    //load segment lengths into registers
    //each thread serially processes a run of segments
    uint32_t threadSegments[NEXT_FIT_SPT];
    LoadSegments(segments, threadSegments, partitionIndex, totalSegCount, totalSegLength);

    //Begin the bin packing
    uint32_t minBinsPacked = 0;  //How many min size bins has this thread packed?
    NextFitBinPack(threadSegments, s_warpHist, minBinsPacked, partitionIndex, totalSegCount);

    //All threads post their count of minimum size bins packed,
    //then participate in an inclusive prefix sum over their block's counts
    s_reduction[threadIdx.x] = InclusiveWarpScan(minBinsPacked);
    __syncthreads();

    if (threadIdx.x < (NEXT_FIT_BLOCK_DIM >> LANE_LOG))
        s_reduction[(threadIdx.x + 1 << LANE_LOG) - 1] = ActiveInclusiveWarpScan(s_reduction[(threadIdx.x + 1 << LANE_LOG) - 1]);
    __syncthreads();

    //Post the status flag for chain scan
    if (!threadIdx.x)
    {
        atomicAdd((uint32_t*)&reduction[partitionIndex],
            (partitionIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | s_reduction[NEXT_FIT_BLOCK_DIM - 1] << 2);
    }

    //Single thread Lookback
    if (partitionIndex)
    {
        if (!threadIdx.x)
        {
            uint32_t prevReduction = 0;
            uint32_t lookBackIndex = partitionIndex - 1;

            while (true)
            {
                const uint32_t flagPayload = reduction[lookBackIndex];
                if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
                {
                    prevReduction += flagPayload >> 2;
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        s_broadcast = prevReduction;
                        atomicAdd((uint32_t*)&reduction[partitionIndex], 1 | (prevReduction << 2));
                        break;
                    }
                    else
                    {
                        lookBackIndex--;
                    }
                }
            }
        }
        __syncthreads();
    }

    //Pass in the reductions from the local and global scans to get the correct
    //offsets to write out the bin information
    const uint32_t prev = (getLaneId() ? s_reduction[threadIdx.x - 1] : 0) +
        (threadIdx.x >= LANE_COUNT ? __shfl_sync(0xffffffff, s_reduction[threadIdx.x - 1], 0) : 0) +
        (partitionIndex ? s_broadcast : 0);

    //Write out the starting offset of each bin and the segment counts of each bin
    for (uint32_t i = 0; i < minBinsPacked; ++i)
    {
        minBinSegCounts[i + prev] = threadSegments[i] & 0xffff;
        binOffsets[i + prev] = (threadSegments[i] >> 16) + NEXT_FIT_BINNING_SIZE * partitionIndex +
            threadIdx.x * NEXT_FIT_SPT;
    }

    //Add the histogram totals from this block for the large segment binning
    if (threadIdx.x < SEG_HIST_SIZE)
        atomicAdd((uint32_t*)&segHist[threadIdx.x], s_hist[threadIdx.x] + s_hist[threadIdx.x + SEG_HIST_SIZE]);
}

//Scan over the histogram
__global__ void SplitSortBinning::Scan(uint32_t* segHist)
{
    uint32_t t = threadIdx.x < SEG_HIST_SIZE ? segHist[threadIdx.x] : 0;
    t = ExclusiveWarpScan(t);
    if (threadIdx.x < SEG_HIST_SIZE)
        segHist[threadIdx.x] = t;
}

//Atomically add to histogram to bin large size segment offsets
__global__ void SplitSortBinning::Bin(
    const uint32_t* segments,
    uint32_t* segHist,
    uint32_t* binOffsets,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < totalSegCount)
    {
        const uint32_t upper = idx == totalSegCount - 1 ? totalSegLength : segments[idx + 1];
        const uint32_t segLength = upper - segments[idx];
        uint32_t position;

        if (MIN_BIN_SIZE < segLength)
        {
            if (segLength <= 64)
                position = atomicAdd((uint32_t*)&segHist[1], 1);
            if (64 < segLength && segLength <= 128)
                position = atomicAdd((uint32_t*)&segHist[2], 1);
            if (128 < segLength && segLength <= 256)
                position = atomicAdd((uint32_t*)&segHist[3], 1);
            if (256 < segLength && segLength <= 512)
                position = atomicAdd((uint32_t*)&segHist[4], 1);
            if (512 < segLength && segLength <= 1024)
                position = atomicAdd((uint32_t*)&segHist[5], 1);
            if (1024 < segLength && segLength <= 2048)
                position = atomicAdd((uint32_t*)&segHist[6], 1);
            if (2048 < segLength && segLength <= 4096)
                position = atomicAdd((uint32_t*)&segHist[7], 1);
            if (4096 < segLength)
                position = atomicAdd((uint32_t*)&segHist[8], 1);

            binOffsets[position] = idx;
        }
    }
}