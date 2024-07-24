/******************************************************************************
 * GPUSorting
 * SplitSort
 * Experimental Hybrid Radix-Merge based SegmentedSort
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 7/5/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SplitSortUtils.cuh"

 //For the scan
#define FLAG_NOT_READY      0
#define FLAG_REDUCTION      1
#define FLAG_INCLUSIVE      2
#define FLAG_MASK           3

namespace SplitSortInternal
{
    template<uint32_t PART_SIZE, uint32_t SPT>
    __device__ __forceinline__ void LoadFull(
        const uint32_t* segments,
        uint32_t* threadSegments,
        const uint32_t partIndex)
    {
        #pragma unroll
        for (uint32_t i = threadIdx.x * SPT + PART_SIZE * partIndex, k = 0;
            k < SPT;
            ++i, ++k)
        {
            threadSegments[k] = segments[i + 1] - segments[i];
        }
    }

    template<uint32_t PART_SIZE, uint32_t SPT>
    __device__ __forceinline__ void LoadPartial(
        const uint32_t* segments,
        uint32_t* threadSegments,
        const uint32_t partIndex,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        #pragma unroll
        for (uint32_t i = threadIdx.x * SPT + PART_SIZE * partIndex, k = 0;
            k < SPT;
            ++i, ++k)
        {
            if (i < totalSegCount)
            {
                const uint32_t upper = i == totalSegCount - 1 ? totalSegLength : segments[i + 1];
                threadSegments[k] = upper - segments[i];
            }
        }
    }

    template<uint32_t PART_SIZE, uint32_t SPT>
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
            {
                LoadFull<PART_SIZE, SPT>(
                    segments,
                    threadSegments,
                    partIndex);
            }
            else
            {
                LoadPartial<PART_SIZE, SPT>(
                    segments,
                    threadSegments,
                    partIndex,
                    totalSegCount,
                    totalSegLength);
            }
        }
        else
        {
            LoadPartial<PART_SIZE, SPT>(
                segments,
                threadSegments,
                partIndex,
                totalSegCount,
                totalSegLength);
        }
    }

    //Simplest approach, which tries to preserve the locality
    //of the segments to optimize memory accesses.
    template<uint32_t MIN_BIN_SIZE>
    __device__ __forceinline__ void PackBin(
        uint32_t* threadSegments,
        uint32_t* s_warpHist,
        uint32_t& currentBinTotal,
        uint32_t& currentBinCount,
        uint32_t& packedBinCount,
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
            if (4096 < threadSegments[i] && threadSegments[i] <= 6144)
                atomicAdd((uint32_t*)&s_warpHist[8], 1);
            if (6144 < threadSegments[i] && threadSegments[i] <= 8192)
                atomicAdd((uint32_t*)&s_warpHist[9], 1);
            if (8192 < threadSegments[i] && threadSegments[i] <= 16384)
                atomicAdd((uint32_t*)&s_warpHist[10], 1);
            if (16384 < threadSegments[i] && threadSegments[i] <= 32768)
                atomicAdd((uint32_t*)&s_warpHist[11], 1);
            if (32768 < threadSegments[i] && threadSegments[i] <= 65536)
                atomicAdd((uint32_t*)&s_warpHist[12], 1);

            //if a segment is longer than 65536, we also
            //count its length, as we will need it later
            if (65536 < threadSegments[i])
            {
                atomicAdd((uint32_t*)&s_warpHist[13], 1);
                atomicAdd((uint32_t*)&s_warpHist[14], threadSegments[i]);
            }
                
            //End the current bin
            if (currentBinCount)
            {
                reinterpret_cast<uint16_t*>(threadSegments)[packedBinCount * 2] = currentBinCount;
                reinterpret_cast<uint16_t*>(threadSegments)[packedBinCount * 2 + 1] = i - currentBinCount;
                atomicAdd((uint32_t*)&s_warpHist[0], 1);

                packedBinCount++;
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
                reinterpret_cast<uint16_t*>(threadSegments)[packedBinCount * 2] = currentBinCount;
                reinterpret_cast<uint16_t*>(threadSegments)[packedBinCount * 2 + 1] = i - currentBinCount;
                atomicAdd((uint32_t*)&s_warpHist[0], 1);

                packedBinCount++;
                currentBinCount = 1;
                currentBinTotal = threadSegments[i];
            }
        }
    }

    //Is there a "hanging" bin leftover?
    __device__ __forceinline__ void AttemptPackHangingBin(
        uint32_t* threadSegments,
        uint32_t* s_warpHist,
        uint32_t& packedBinCount,
        const uint32_t currentBinCount,
        const uint32_t threadRunLength)
    {
        if (currentBinCount)
        {
            reinterpret_cast<uint16_t*>(threadSegments)[packedBinCount * 2] = currentBinCount;
            reinterpret_cast<uint16_t*>(threadSegments)[packedBinCount * 2 + 1] = threadRunLength - currentBinCount;
            atomicAdd((uint32_t*)&s_warpHist[0], 1);

            packedBinCount++;
        }
    }

    template<uint32_t SPT, uint32_t MIN_BIN_SIZE>
    __device__ __forceinline__ void NextFitBinPackFull(
        uint32_t* threadSegments,
        uint32_t* s_warpHist,
        uint32_t& packedBinCount)
    {
        uint32_t currentBinTotal = 0;   //What is the total length of segments in this current bin
        uint32_t currentBinCount = 0;   //How many segments are in this current bin?

        #pragma unroll
        for (uint32_t i = 0; i < SPT; ++i)
        {
            PackBin<MIN_BIN_SIZE>(
                threadSegments,
                s_warpHist,
                currentBinTotal,
                currentBinCount,
                packedBinCount,
                i);
        }

        AttemptPackHangingBin(
            threadSegments,
            s_warpHist,
            packedBinCount,
            currentBinCount,
            SPT);
    }

    template<int32_t SPT, uint32_t MIN_BIN_SIZE>
    __device__ __forceinline__ void NextFitBinPackPartial(
        uint32_t* threadSegments,
        uint32_t* s_warpHist,
        uint32_t& packedBinCount,
        const uint32_t partitionSize)
    {
        uint32_t currentBinTotal = 0;   //What is the total length of segments in this current bin
        uint32_t currentBinCount = 0;   //How many segments are in this current bin?

        int32_t threadRunLength = partitionSize - threadIdx.x * SPT;
        if (threadRunLength > SPT)
            threadRunLength = SPT;
        if (threadRunLength < 0)
            threadRunLength = 0;

        for (uint32_t i = 0; i < (uint32_t)threadRunLength; ++i)
        {
            PackBin<MIN_BIN_SIZE>(
                threadSegments,
                s_warpHist,
                currentBinTotal,
                currentBinCount,
                packedBinCount,
                i);
        }

        AttemptPackHangingBin(
            threadSegments,
            s_warpHist,
            packedBinCount,
            currentBinCount,
            (uint32_t)threadRunLength);
    }

    template<
        uint32_t PART_SIZE,
        uint32_t SPT,
        uint32_t MIN_BIN_SIZE>
    __device__ __forceinline__ void NextFitBinPack(
        uint32_t* threadSegments,
        uint32_t* s_warpHist,
        uint32_t& packedBinCount,
        const uint32_t partIndex,
        const uint32_t totalSegCount)
    {
        if (partIndex < gridDim.x - 1)
        {
            NextFitBinPackFull<SPT, MIN_BIN_SIZE>(
                threadSegments,
                s_warpHist,
                packedBinCount);
        }
        
        if(partIndex == gridDim.x - 1)
        {
            NextFitBinPackPartial<(int32_t)SPT, MIN_BIN_SIZE>(
                threadSegments,
                s_warpHist,
                packedBinCount,
                totalSegCount - PART_SIZE * partIndex);
        }
    }

    template<uint32_t WARPS>
    __device__ __forceinline__ uint32_t ScanAndPostStatusFlag(
        volatile uint32_t* reduction,
        uint32_t* s_reduction,
        const uint32_t toScan,
        const uint32_t partitionIndex)
    {
        uint32_t warpScan = InclusiveWarpScanCircularShift(toScan);
        if (!getLaneId())
            s_reduction[WARP_INDEX] = warpScan;
        __syncthreads();

        if (threadIdx.x < LANE_COUNT)
        {
            const bool p = threadIdx.x < WARPS;
            const uint32_t t = InclusiveWarpScanCircularShift(p ? s_reduction[threadIdx.x] : 0);
            if (p)
                s_reduction[threadIdx.x] = t;
        }
        __syncthreads();

        //Post the status flag for chain scan
        if (!threadIdx.x)
        {
            atomicAdd((uint32_t*)&reduction[partitionIndex],
                (partitionIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | s_reduction[0] << 2);
        }

        return warpScan;
    }

    __device__ __forceinline__ void Lookback(
        volatile uint32_t* reduction,
        uint32_t& s_broadcast,
        const uint32_t& partitionIndex)
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

    //This kernel has 3 jobs:
    //1)Pack segments of length <= 32 into bins
    //2)Bin segments > 32 using Hou style approach
    //3)Sum all segments > 65536, if any, to be used later
    template<
        uint32_t PART_SIZE,     //Size of a partition tile
        uint32_t WARPS,         //Warps in a threadblock
        uint32_t SPT,           //Segments per thread
        uint32_t MIN_BIN_SIZE,  //Segments below this length are packed together
        uint32_t SEG_INFO_SIZE> //Size of the segment hist
    __global__ void NextFitBinPacking(
        const uint32_t* segments,
        uint32_t* segHist,
        uint32_t* packedSegCounts,
        uint32_t* binOffsets,
        volatile uint32_t* index,
        volatile uint32_t* reduction,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        __shared__ uint32_t s_hist[SEG_INFO_SIZE * WARPS];
        __shared__ uint32_t s_reduction[WARPS];
        __shared__ uint32_t s_broadcast;

        uint32_t* s_warpHist = &s_hist[SEG_INFO_SIZE * WARP_INDEX];
        if (getLaneId() < SEG_INFO_SIZE)
            s_warpHist[getLaneId()] = 0;

        //do the chained scan thing
        if (!threadIdx.x)
            s_broadcast = atomicAdd((uint32_t*)&index[0], 1);
        __syncthreads();
        const uint32_t partitionIndex = s_broadcast;

        //load segment lengths into registers
        //each thread serially processes a run of segments
        uint32_t threadSegments[SPT];
        LoadSegments<PART_SIZE, SPT>(
            segments,
            threadSegments,
            partitionIndex,
            totalSegCount,
            totalSegLength);

        //Begin the bin packing
        uint32_t packedBinCount = 0;  //How many packed bins (bins with more than one segment) has this thread processed?  
        NextFitBinPack<PART_SIZE, SPT, MIN_BIN_SIZE>(
            threadSegments,
            s_warpHist,
            packedBinCount,
            partitionIndex,
            totalSegCount);

        //Device wide prefix sum of packed bins count
        const uint32_t warpScan = ScanAndPostStatusFlag<WARPS>(
            reduction,
            s_reduction,
            packedBinCount,
            partitionIndex);

        if (partitionIndex)
            Lookback(reduction, s_broadcast, partitionIndex);

        //warp + block + device 
        const uint32_t prev = (getLaneId() ? warpScan : 0) + (threadIdx.x >= LANE_COUNT ? s_reduction[WARP_INDEX] : 0) +
            (partitionIndex ? s_broadcast : 0);

        //Write out the starting offset of each bin and the segment counts of each bin
        for (uint32_t i = 0; i < packedBinCount; ++i)
        {
            packedSegCounts[i + prev] = threadSegments[i] & 0xffff;
            binOffsets[i + prev] = (threadSegments[i] >> 16) + PART_SIZE * partitionIndex +
                threadIdx.x * SPT;
        }

        //Add the histogram totals from this block for the large segment binning
        if (threadIdx.x < SEG_INFO_SIZE)
            atomicAdd((uint32_t*)&segHist[threadIdx.x], s_hist[threadIdx.x] + s_hist[threadIdx.x + SEG_INFO_SIZE]);
    }

    //Scan over the histogram
    template<uint32_t SEG_HIST_SIZE>
    __global__ void BinningScan(uint32_t* segHist)
    {
        const bool p = threadIdx.x < SEG_HIST_SIZE;
        const uint32_t t = InclusiveWarpScanCircularShift(p ? segHist[threadIdx.x] : 0);
        if (p)
            segHist[threadIdx.x] = t;
    }

    template<uint32_t MIN_BIN_SIZE>
    __device__ __forceinline__ void Bin(
        const uint32_t& segLength,
        uint32_t* segHist,
        uint32_t* binOffsets,
        uint32_t segIndex)
    {
        if (MIN_BIN_SIZE < segLength)
        {
            uint32_t position;

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
            if (4096 < segLength && segLength <= 6144)
                position = atomicAdd((uint32_t*)&segHist[8], 1);
            if (6144 < segLength && segLength <= 8192)
                position = atomicAdd((uint32_t*)&segHist[9], 1);
            if (8192 < segLength && segLength <= 16384)
                position = atomicAdd((uint32_t*)&segHist[10], 1);
            if (16384 < segLength && segLength <= 32768)
                position = atomicAdd((uint32_t*)&segHist[11], 1);
            if (32768 < segLength && segLength <= 65536)
                position = atomicAdd((uint32_t*)&segHist[12], 1);

            binOffsets[position] = segIndex;
        }
    }

    //Atomically add to histogram to bin large size segment offsets
    template<uint32_t MIN_BIN_SIZE>
    __global__ void BinSimple(
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
            Bin<MIN_BIN_SIZE>(
                segLength,
                segHist,
                binOffsets,
                idx);
        }
    }

    __device__ __forceinline__ void ThreadScan(
        uint32_t& segLength,
        uint32_t& reduction)
    {
        if (segLength <= 65536)
            segLength = 0;

        const uint32_t t = segLength;
        segLength = reduction;
        reduction += t;
    }
    
    //This kernel has 2 jobs:
    //1) bin segment lengths > 32
    //2) Track the offsets necessary to create a contiguous buffer
    //from the segments of length > 65536
    template<
        uint32_t PART_SIZE,     //size of a partition tile
        uint32_t SPT,           //segments per thread
        uint32_t WARPS,         //warps in a threadblock
        uint32_t MIN_BIN_SIZE>  //Segments below this length are packed together
    __global__ void BinAndCoalesce(
        const uint32_t* segments,
        uint32_t* segHist,
        uint32_t* binOffsets,
        uint32_t* largeSegmentOffsets,
        volatile uint32_t* index,
        volatile uint32_t* reduction,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        __shared__ uint32_t s_broadcast;
        __shared__ uint32_t s_reduction[WARPS];

        //do the chained scan thing
        if (!threadIdx.x)
            s_broadcast = atomicAdd((uint32_t*)&index[0], 1);
        __syncthreads();
        const uint32_t partitionIndex = s_broadcast;

        uint32_t threadSegments[SPT];
        LoadSegments<PART_SIZE, SPT>(
            segments,
            threadSegments,
            partitionIndex,
            totalSegCount,
            totalSegLength);

        uint32_t threadReduction = 0;
        if (partitionIndex < gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x * SPT + partitionIndex * PART_SIZE, k = 0;
                k < SPT;
                ++i, ++k)
            {
                Bin<MIN_BIN_SIZE>(
                    threadSegments[k],
                    segHist,
                    binOffsets,
                    i);

                ThreadScan(threadSegments[k], threadReduction);
            }
        }

        if (partitionIndex == gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x * SPT + partitionIndex * PART_SIZE, k = 0;
                k < SPT;
                ++i, ++k)
            {
                if (i < totalSegCount)
                {
                    Bin<MIN_BIN_SIZE>(
                        threadSegments[k],
                        segHist,
                        binOffsets,
                        i);

                    ThreadScan(threadSegments[k], threadReduction);
                }
            }
        }

        //Device wide prefix sum of segment lengths whose length > 65536
        const uint32_t warpScan = ScanAndPostStatusFlag<WARPS>(
            reduction,
            s_reduction,
            threadReduction,
            partitionIndex);

        if (partitionIndex)
            Lookback(reduction, s_broadcast, partitionIndex);

        //warp + block + device 
        const uint32_t prev = (getLaneId() ? warpScan : 0) + (threadIdx.x >= LANE_COUNT ? s_reduction[WARP_INDEX] : 0) + 
            (partitionIndex ? s_broadcast : 0);

        if (partitionIndex < gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x * SPT + partitionIndex * PART_SIZE, k = 0;
                k < SPT;
                ++i, ++k)
            {
                largeSegmentOffsets[i] = (k ? threadSegments[k] : 0) + prev;
            }
        }

        if (partitionIndex == gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x * SPT + partitionIndex * PART_SIZE, k = 0;
                k < SPT;
                ++i, ++k)
            {
                if(i < totalSegCount)
                    largeSegmentOffsets[i] = (k ? threadSegments[k] : 0) + prev;
            }
        }
    }
}

#undef FLAG_MASK
#undef FLAG_INCLUSIVE
#undef FLAG_REDUCTION
#undef FLAG_NOT_READY