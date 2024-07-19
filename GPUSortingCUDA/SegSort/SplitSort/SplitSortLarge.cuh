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

#define FLAG_NOT_READY      0
#define FLAG_REDUCTION      1
#define FLAG_INCLUSIVE      2
#define FLAG_MASK           3

#define RADIX               256
#define RADIX_MASK          255
#define RADIX_LOG           8

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl;}

namespace SplitSortInternal
{
    //***********************************************************************
    //SPLIT SORT LARGE DEVICE FUNCTIONS
    //***********************************************************************
    template<uint32_t HIST_BYTES>
    __device__ __forceinline__  void count(
        const uint8_t* key8,
        uint32_t* s_hist0,
        uint32_t* s_hist1,
        uint32_t* s_hist2,
        uint32_t* s_hist3)
    {
        //Will never happen
        //TODO error assertion here or something to be safe ?
    }

    template<>
    __device__ __forceinline__ void count<1>(
        const uint8_t* key8,
        uint32_t* s_warpsHist0,
        uint32_t* s_warpsHist1,
        uint32_t* s_warpsHist2,
        uint32_t* s_warpsHist3)
    {
        atomicAdd(&s_warpsHist0[key8[0]], 1);
    }

    template<>
    __device__ __forceinline__ void count<2>(
        const uint8_t* key8,
        uint32_t* s_warpsHist0,
        uint32_t* s_warpsHist1,
        uint32_t* s_warpsHist2,
        uint32_t* s_warpsHist3)
    {
        atomicAdd(&s_warpsHist0[key8[0]], 1);
        atomicAdd(&s_warpsHist1[key8[1]], 1);
    }

    template<>
    __device__ __forceinline__ void count<3>(
        const uint8_t* key8,
        uint32_t* s_warpsHist0,
        uint32_t* s_warpsHist1,
        uint32_t* s_warpsHist2,
        uint32_t* s_warpsHist3)
    {
        atomicAdd(&s_warpsHist0[key8[0]], 1);
        atomicAdd(&s_warpsHist1[key8[1]], 1);
        atomicAdd(&s_warpsHist2[key8[2]], 1);
    }

    template<>
    __device__ __forceinline__ void count<4>(
        const uint8_t* key8,
        uint32_t* s_warpsHist0,
        uint32_t* s_warpsHist1,
        uint32_t* s_warpsHist2,
        uint32_t* s_warpsHist3)
    {
        atomicAdd(&s_warpsHist0[key8[0]], 1);
        atomicAdd(&s_warpsHist1[key8[1]], 1);
        atomicAdd(&s_warpsHist2[key8[2]], 1);
        atomicAdd(&s_warpsHist3[key8[3]], 1);
    }

    __device__ __forceinline__ void ReduceAndAddDigitCounts(
        uint32_t* globalHistogram,
        uint32_t* s_hist0,
        uint32_t* s_hist1,
        uint32_t* s_hist2,
        uint32_t* s_hist3)
    {
        constexpr uint32_t RADIX_OFFSET_SEC = RADIX;
        constexpr uint32_t RADIX_OFFSET_THIRD = RADIX * 2;
        constexpr uint32_t RADIX_OFFSET_FOURTH = RADIX * 3;
        for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
        {
            atomicAdd(&globalHistogram[i], s_hist0[i] + s_hist0[i + RADIX]);
            atomicAdd(&globalHistogram[i + RADIX_OFFSET_SEC], s_hist1[i] + s_hist1[i + RADIX]);
            atomicAdd(&globalHistogram[i + RADIX_OFFSET_THIRD], s_hist2[i] + s_hist2[i + RADIX]);
            atomicAdd(&globalHistogram[i + RADIX_OFFSET_FOURTH], s_hist3[i] + s_hist3[i + RADIX]);
        }
    }

    //We already have the seglength, which is the digit count 
    //of segIds. So we add this directly to the global hist
    template<uint32_t FIX_BYTES>
    __device__ __forceinline__ void AddSegIdDigitCounts(
        uint32_t* globalHistogram,
        const uint32_t binOffset,
        const uint32_t totalLocalLength)
    {
        uint32_t segId[1] = { binOffset };
        uint8_t* segId8 = reinterpret_cast<uint8_t*>(segId);

        if (!threadIdx.x)
        {
            constexpr uint32_t RADIX_OFFSET_FIFTH = RADIX * 4;
            atomicAdd((uint32_t*)&globalHistogram[segId8[0] + RADIX_OFFSET_FIFTH], totalLocalLength);

            if constexpr (FIX_BYTES > 1)
            {
                constexpr uint32_t RADIX_OFFSET_SIXTH = RADIX * 5;
                atomicAdd(&globalHistogram[segId8[1] + RADIX_OFFSET_SIXTH], totalLocalLength);
            }

            if constexpr (FIX_BYTES > 2)
            {
                constexpr uint32_t RADIX_OFFSET_SEVENTH = RADIX * 6;
                atomicAdd(&globalHistogram[segId8[1] + RADIX_OFFSET_SEVENTH], totalLocalLength);
            }

            if constexpr (FIX_BYTES > 3)
            {
                constexpr uint32_t RADIX_OFFSET_EIGTH = RADIX * 7;
                atomicAdd(&globalHistogram[segId8[1] + RADIX_OFFSET_EIGTH], totalLocalLength);
            }
        }
    }

    //This kernel has 3 jobs:
    //1) histogram digit counts
    //2) pack or store segIds to be used in radix passes
    //3) gather the keys and place them into a contiguous second buffer
    template<
        class V,
        class F,
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        uint32_t RADIX_PASSES>
    __global__ void GlobalHistAndFix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* largeSegmentOffsets,
        const uint32_t* sort,
        const V* values,
        uint32_t* altA,
        V* altPayloadA,
        F* fixA,
        uint32_t* globalHistogram,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t size)
    {
        constexpr uint32_t sMemSize = WARPS / 2 * RADIX;
        __shared__ uint32_t s_hist0[sMemSize];
        __shared__ uint32_t s_hist1[sMemSize];
        __shared__ uint32_t s_hist2[sMemSize];
        __shared__ uint32_t s_hist3[sMemSize];

        //64 threads : 1 histogram in shared memory
        uint32_t* s_warpsHist0 = &s_hist0[threadIdx.x / 64 * RADIX];
        uint32_t* s_warpsHist1 = &s_hist1[threadIdx.x / 64 * RADIX];
        uint32_t* s_warpsHist2 = &s_hist2[threadIdx.x / 64 * RADIX];
        uint32_t* s_warpsHist3 = &s_hist3[threadIdx.x / 64 * RADIX];

        //Advance pointers to segment start
        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t coalescedStart = largeSegmentOffsets[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        values += segmentStart;
        altA += coalescedStart;
        altPayloadA += coalescedStart;
        fixA += coalescedStart;
        
        //clear
        for (uint32_t i = threadIdx.x; i < sMemSize; i += blockDim.x)
        {
            s_hist0[i] = 0;
            s_hist1[i] = 0;
            s_hist2[i] = 0;
            s_hist3[i] = 0;
        }
        __syncthreads();

        //If the number of passes is less than 4, we pack the segIds into the key
        //else, a seperate buffer holds the value
        if constexpr (RADIX_PASSES <= 4)
        {
            for (uint32_t i = threadIdx.x; i < totalLocalLength; i += blockDim.x)
            {
                uint32_t key[1] = { sort[i] };
                if constexpr (BITS_TO_SORT < 32)   //Dumb hack to supress compiler warning. This is already checked
                    key[0] |= binOffset << BITS_TO_SORT;
                count<RADIX_PASSES>(
                    reinterpret_cast<uint8_t*>(key),
                    s_warpsHist0,
                    s_warpsHist1,
                    s_warpsHist2,
                    s_warpsHist3);
                altA[i] = key[0];
                altPayloadA[i] = values[i];
            }
        }
        else
        {
            for (uint32_t i = threadIdx.x; i < totalLocalLength; i += blockDim.x)
            {
                uint32_t key[1] = { sort[i] };
                count<4>(
                    reinterpret_cast<uint8_t*>(key),
                    s_warpsHist0,
                    s_warpsHist1,
                    s_warpsHist2,
                    s_warpsHist3);
                altA[i] = key[0];
                fixA[i] = binOffset;
                altPayloadA[i] = values[i];
            }
        }
        __syncthreads();

        ReduceAndAddDigitCounts(
            globalHistogram,
            s_hist0,
            s_hist1,
            s_hist2,
            s_hist3);
            
        //if the number of radix passes is greater than 4
        //then the segId's need to be histogrammed seperately.
        if constexpr (RADIX_PASSES > 4)
        {
            AddSegIdDigitCounts<sizeof(F)>(
                globalHistogram,
                binOffset,
                totalLocalLength);
        }
    }

    __global__ void LargeScan(
        uint32_t* globalHistogram,
        uint32_t* passHistograms,
        const uint32_t passPartitions)
    {
        __shared__ uint32_t s_scan[RADIX >> LANE_LOG];

        uint32_t scan = InclusiveWarpScanCircularShift(globalHistogram[threadIdx.x + blockIdx.x * RADIX]);
        if (!getLaneId())
            s_scan[WARP_INDEX] = scan;
        __syncthreads();

        if (threadIdx.x < LANE_COUNT)
        {
            const bool p = threadIdx.x < (RADIX >> LANE_LOG);
            const uint32_t t = ExclusiveWarpScan(p ? s_scan[threadIdx.x] : 0);
            if (p)
                s_scan[threadIdx.x] = t;
        }
        __syncthreads();

        passHistograms[threadIdx.x + passPartitions * RADIX * blockIdx.x] =
            ((getLaneId() ? scan : 0) + s_scan[WARP_INDEX]) << 2 | FLAG_INCLUSIVE;
    }

    //Deprecate these? Massively increases reg counts for some reason
    template<
        class P,
        uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void ScatterNonKey(
        const P* payloadSource,
        P* payloadDest,
        const uint32_t* digits,
        const uint32_t* offsets,
        const uint32_t* s_localHist,
        P* s_mem)
    {
        //Load into registers
        P payloads[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
            payloads[k] = payloadSource[i];

        //Scatter into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
            s_mem[offsets[i]] = payloads[i];
        __syncthreads();

        //Scatter from shared into device
        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            payloadDest[s_localHist[digits[k]] + i] = s_mem[i];
    }

    template<
        class P,
        uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void ScatterNonKeyPartial(
        const P* payloadSource,
        P* payloadDest,
        const uint32_t* digits,
        const uint32_t* offsets,
        const uint32_t* s_localHist,
        P* s_mem,
        const uint32_t finalPartSize)
    {
        __syncthreads();

        //Load into registers
        P payloads[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
        {
            if(i < finalPartSize)
                payloads[k] = payloadSource[i];
        }

        //Scatter into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
            s_mem[offsets[i]] = payloads[i];
        __syncthreads();

        //Scatter from shared into device
        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if(i < finalPartSize)
                payloadDest[s_localHist[digits[k]] + i] = s_mem[i];
        }
    }

    template<uint32_t KEYS_PER_THREAD, uint32_t WARPS>
    __device__ __forceinline__ uint32_t LoadOffset(const uint32_t partitionIndex)
    {
        constexpr uint32_t SUB_PART_SIZE = KEYS_PER_THREAD * LANE_COUNT;
        constexpr uint32_t PART_SIZE = SUB_PART_SIZE * WARPS;
        return getLaneId() + WARP_INDEX * SUB_PART_SIZE + partitionIndex * PART_SIZE;
    }

    //TODO Reuse as much as possible from SplitSort Variants
    template<
        class K,
        class V,
        class F,
        uint32_t WARPS,
        uint32_t KEYS_PER_THREAD,
        uint32_t RADIX_SHIFT,
        bool IS_NOT_PACKED>
    __global__ void DigitBinningPass(
        const K* altA,
        const V* altPayloadA,
        const F* fixA,
        K* altB,
        V* altPayloadB,
        F* fixB,
        volatile uint32_t* passHistogram,
        volatile uint32_t* index,
        const uint32_t size)
    {
        constexpr uint32_t HIST_BYTES = WARPS * RADIX * sizeof(uint32_t);
        constexpr uint32_t PART_BYTES = KEYS_PER_THREAD * LANE_COUNT * WARPS * sizeof(V); //assume V >= K
        constexpr uint32_t SMEM_BYTES = HIST_BYTES > PART_BYTES ? HIST_BYTES : PART_BYTES;
        __shared__ uint32_t s_mem[SMEM_BYTES / sizeof(uint32_t)];
        __shared__ uint32_t s_localHistogram[RADIX];
        __shared__ uint32_t s_broadcast;
        uint32_t* s_warpHist = &s_mem[WARP_INDEX << RADIX_LOG];

        //clear
        for (uint32_t i = getLaneId(); i < RADIX; i += LANE_COUNT)
            s_warpHist[i] = 0;

        //atomically assign partition tiles
        if (threadIdx.x == 0)
            s_broadcast = atomicAdd((uint32_t*)&index[0], 1);
        __syncthreads();
        const uint32_t partitionIndex = s_broadcast;

        //load keys
        uint32_t keys[KEYS_PER_THREAD];
        if (partitionIndex < gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = LoadOffset<KEYS_PER_THREAD, WARPS>(partitionIndex), k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                keys[k] = (uint32_t)altA[i];
            }
        }

        if (partitionIndex == gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = LoadOffset<KEYS_PER_THREAD, WARPS>(partitionIndex), k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                keys[k] = i < size ? (uint32_t)altA[i] : 0xffffffff;
            }
        }

        uint32_t offsets[KEYS_PER_THREAD];
        /*
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
        {
            uint32_t eqMask = 0xffffffff;
            #pragma unroll
            for (int32_t bit = 0; bit < RADIX_LOG; ++bit)
            {
                uint32_t current_bit = 1 << bit + RADIX_SHIFT;
                asm("{\n"
                    "    .reg .pred p;\n"
                    "    .reg .b32 bal;\n"
                    "    and.b32 bal, %1, %2;\n"
                    "    setp.ne.u32 p, bal, 0;\n"
                    "    vote.ballot.sync.b32 bal, p, 0xffffffff;\n"
                    "    @!p not.b32 bal, bal;\n"
                    "    and.b32 %0, %0, bal;\n"
                    "}\n" : "+r"(eqMask) : "r"(keys[i]), "r"(current_bit));
            }
            offsets[i] = __popc(eqMask & getLaneMaskLt());
            const uint32_t highestRankPeer = LANE_COUNT - __clz(eqMask) - 1;
            uint32_t preIncrementVal;
            if (getLaneId() == highestRankPeer)
                preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> RADIX_SHIFT & RADIX_MASK], offsets[i] + 1);

            offsets[i] += __shfl_sync(0xffffffff, preIncrementVal, highestRankPeer);
        }
        */
        RankKeys<KEYS_PER_THREAD, RADIX_LOG, RADIX_MASK>(
            keys,
            offsets,
            s_warpHist,
            RADIX_SHIFT);
        __syncthreads();

        if (threadIdx.x < RADIX)
        {
            uint32_t reduction = s_mem[threadIdx.x];
            constexpr uint32_t HISTS_SIZE = WARPS - 1;
            #pragma unroll
            for (uint32_t i = threadIdx.x + RADIX, k = 0; k < HISTS_SIZE; i += RADIX, ++k)
            {
                reduction += s_mem[i];
                s_mem[i] = reduction - s_mem[i];
            }

            if (partitionIndex < gridDim.x - 1)
            {
                atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
                    FLAG_REDUCTION | reduction << 2);
            }
            
            s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT)
        {
            const bool p = threadIdx.x < (RADIX >> LANE_LOG);
            const uint32_t t = ExclusiveWarpScan(p ? s_localHistogram[threadIdx.x << LANE_LOG] : 0);
            if (p)
                s_localHistogram[threadIdx.x << LANE_LOG] = t;
        }
        __syncthreads();

        if (threadIdx.x < RADIX && getLaneId())
            s_localHistogram[threadIdx.x] += __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
        __syncthreads();

        if (WARP_INDEX)
        {
            #pragma unroll 
            for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
            {
                const uint32_t t2 = keys[i] >> RADIX_SHIFT & RADIX_MASK;
                offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
            }
        }
        else
        {
            #pragma unroll
            for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
                offsets[i] += s_localHistogram[keys[i] >> RADIX_SHIFT & RADIX_MASK];
        }
        __syncthreads();

        //lookback
        if (threadIdx.x < RADIX)
        {
            uint32_t reduction = 0;
            uint32_t lookbackIndex = threadIdx.x + partitionIndex * RADIX;
            while (true)
            {
                const uint32_t flagPayload = passHistogram[lookbackIndex];

                if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    reduction += flagPayload >> 2;
                    if (partitionIndex < gridDim.x - 1)
                    {
                        atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
                            1 | (reduction << 2));
                    }
                    s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
                    break;
                }

                if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
                {
                    reduction += flagPayload >> 2;
                    lookbackIndex -= RADIX;
                }
            }
        }

        //scatter keys into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
            s_mem[offsets[i]] = keys[i];
        __syncthreads();

        uint32_t digits[KEYS_PER_THREAD];
        const uint32_t finalPartSize = size - partitionIndex * KEYS_PER_THREAD * blockDim.x;
        //store the digit of key in register
        #pragma unroll
        for (uint32_t k = 0, t = threadIdx.x; k < KEYS_PER_THREAD;
            ++k, t += blockDim.x)
        {
            if (t < finalPartSize)
            {
                digits[k] = s_mem[t] >> RADIX_SHIFT & RADIX_MASK;
                altB[s_localHistogram[digits[k]] + t] = (K)s_mem[t];
            }
        }
        __syncthreads();

        //Load into registers
        V payloads[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = LoadOffset<KEYS_PER_THREAD, WARPS>(partitionIndex), k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if(i < size)
                payloads[k] = altPayloadA[i];
        }

        //Scatter into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
            reinterpret_cast<V*>(s_mem)[offsets[i]] = payloads[i];
        __syncthreads();

        //Scatter from shared into device
        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if(i < finalPartSize)
                altPayloadB[s_localHistogram[digits[k]] + i] = reinterpret_cast<V*>(s_mem)[i];
        }

        if constexpr (IS_NOT_PACKED)
        {
            __syncthreads();

            F fixes[KEYS_PER_THREAD];
            #pragma unroll
            for (uint32_t i = LoadOffset<KEYS_PER_THREAD, WARPS>(partitionIndex), k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                if(i < size)
                    fixes[k] = fixA[i];
            }

            //Scatter into shared memory
            #pragma unroll
            for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
                reinterpret_cast<F*>(s_mem)[offsets[i]] = fixes[i];
            __syncthreads();

            //Scatter from shared into device
            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if(i < finalPartSize)
                    fixB[s_localHistogram[digits[k]] + i] = reinterpret_cast<F*>(s_mem)[i];
            }
        }
    }

    //Scatter the sorted keys back to their respective segments
    //Remove the packed bits if necessary
    template<class V, class F, uint32_t PART_SIZE, uint32_t KEYS_PER_THREAD>
    __global__ void ScatterBack(
        const uint32_t* segments,
        const uint32_t* largeSegmentOffsets,
        const uint32_t* alt,
        const V* altPayloads,
        const F* fix,
        uint32_t* sort,
        V* values,
        const uint32_t size,
        const uint32_t totalSegLength)
    {
        //How many different segments is it possible to encounter?
        //constexpr uint32_t bigSize = 8192 > PART_SIZE ? 8192 : PART_SIZE;
        //constexpr uint32_t littleSize = 8192 <= PART_SIZE ? 8192 : PART_SIZE;
        //constexpr uint32_t possibleLocations = (bigSize + littleSize - 1) / littleSize + 1;
        uint32_t largeSeg[KEYS_PER_THREAD];
        uint32_t segs[KEYS_PER_THREAD];
        
        uint32_t fixes[KEYS_PER_THREAD];
        uint32_t keys[KEYS_PER_THREAD];
        V payloads[KEYS_PER_THREAD];

        if (blockIdx.x < gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x + blockIdx.x * PART_SIZE, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                fixes[k] = (uint32_t)fix[i];
                keys[k] = alt[i];
                payloads[k] = altPayloads[i];
            }
        }

        if (blockIdx.x == gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x + blockIdx.x * PART_SIZE, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (i < size)
                {
                    fixes[k] = (uint32_t)fix[i];
                    keys[k] = alt[i];
                    payloads[k] = altPayloads[i];
                }
                else
                {
                    fixes[k] = 0;
                }
            }
        }
        
        //Load segment offsets
        uint32_t prev = fixes[0];
        uint32_t tOffset = 1;
        largeSeg[0] = largeSegmentOffsets[fixes[0]];
        segs[0] = segments[fixes[0]];
        #pragma unroll
        for (uint32_t i = 1; i < KEYS_PER_THREAD; ++i)
        {
            if (prev != fixes[i])
            {
                prev = fixes[i];
                largeSeg[tOffset] = largeSegmentOffsets[fixes[i]];
                segs[tOffset] = segments[fixes[i]];
                tOffset++;
            }
        }

        tOffset = 0;
        prev = fixes[0];
        if (blockIdx.x < gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x + blockIdx.x * PART_SIZE, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (k && prev != fixes[k])
                {
                    prev = fixes[k];
                    tOffset++;
                }
                const uint32_t scatterTo = i - largeSeg[tOffset] + segs[tOffset];
                sort[scatterTo] = keys[k];
                values[scatterTo] = payloads[k];
            }
        }

        if (blockIdx.x == gridDim.x - 1)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x + blockIdx.x * PART_SIZE, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (i < size)
                {
                    if (k && prev != fixes[k])
                    {
                        prev = fixes[k];
                        tOffset++;
                    }

                    const uint32_t scatterTo = i - largeSeg[tOffset] + segs[tOffset];
                    sort[scatterTo] = keys[k];
                    values[scatterTo] = payloads[k];
                }
            }
        }
    }

    //***********************************************************************
    //SPLIT SORT LARGE HOST FUNCTIONS
    //***********************************************************************
    
    //We want: 
    //1) The number of radixPasses to be compile time visible
    //2) To vary the number of bytes used for the fix buffer, if any
    //so we wrap templates of the dispatch in a switch statement. Maybe there
    //is a better way to do this?
    template<class V, uint32_t BITS_TO_SORT>
    __host__ void DispatchGlobalHistAndFix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* largeSegmentOffsets,
        uint32_t* sort,
        V* values,
        uint32_t* altA,
        V* altPayloadA,
        uint32_t* fixA,
        uint32_t* globalHistogram,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segsInCurBin,
        const uint32_t radixPasses,
        const uint32_t size)
    {
        switch (radixPasses)
        {
        case 1:
            GlobalHistAndFix<V, uint32_t, 4, BITS_TO_SORT, 1><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                fixA,
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 2:
            GlobalHistAndFix<V, uint32_t, 4, BITS_TO_SORT, 2><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                fixA,
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 3:
            GlobalHistAndFix<V, uint32_t, 4, BITS_TO_SORT, 3><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                fixA,
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 4:
            GlobalHistAndFix<V, uint32_t, 4, BITS_TO_SORT, 4><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                fixA,
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 5:
            GlobalHistAndFix<V, uint8_t, 4, BITS_TO_SORT, 5><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                reinterpret_cast<uint8_t*>(fixA),
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 6:
            GlobalHistAndFix<V, uint16_t, 4, BITS_TO_SORT, 6><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                reinterpret_cast<uint16_t*>(fixA),
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 7:
            GlobalHistAndFix<V, uint32_t, 4, BITS_TO_SORT, 7><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                fixA,
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        case 8:
            GlobalHistAndFix<V, uint32_t, 4, BITS_TO_SORT, 8><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                largeSegmentOffsets,
                sort,
                values,
                altA,
                altPayloadA,
                fixA,
                globalHistogram,
                totalSegCount,
                totalSegLength,
                size);
            break;
        default:
            break;
        }
    }

    template<class V, uint32_t BITS_TO_SORT>
    __host__ void SplitSortLarge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* largeSegmentOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segsInCurBin,
        const uint32_t size)
    {
        const uint32_t k_warps = 16;            //<----------These are the main tuning parameters
        const uint32_t k_keysPerThread = 15;    //<----------
        const uint32_t k_partitionSize = k_keysPerThread * LANE_COUNT * k_warps;

        const uint32_t segIdBits = findHighestBit(totalSegCount);
        const uint32_t totalBits = segIdBits + BITS_TO_SORT;
        const uint32_t radixPasses = totalBits <= 32 ?
            dvrup<RADIX_LOG>(totalBits) : 4 + dvrup<RADIX_LOG>(segIdBits);
        const uint32_t binningPartitions = (size + k_partitionSize - 1) / k_partitionSize;
        uint32_t fixBytes;
        if (totalBits <= 32)
            fixBytes = 0;
        else if (segIdBits <= 8)
            fixBytes = sizeof(uint8_t);
        else if (segIdBits <= 16)
            fixBytes = sizeof(uint16_t);
        else
            fixBytes = sizeof(uint32_t);

        //temporary memory
        void* temp;
        //TODO CUDA ERRCHECK
        cudaMalloc(&temp,
            (size * sizeof(V) * 2) +                    //Alt payload
            (size * sizeof(uint32_t) * 2) +             //Alt keys
            (size * fixBytes * 2) +                     //Fix buffer, if necessary
            ((radixPasses +                             //Chained scan atomic bump
            (radixPasses * RADIX) +                     //GlobalHistogram
            (binningPartitions * radixPasses * RADIX))  //PassHistograms
            * sizeof(uint32_t)));

        V* altPayloadA = (V*)temp;
        V* altPayloadB = &((V*)temp)[size];
        uint32_t offset = size * 2 * (sizeof(V) / sizeof(uint32_t));
        uint32_t* altA = &((uint32_t*)temp)[offset];
        offset += size;
        uint32_t* altB = &((uint32_t*)temp)[offset];
        offset += size;
        uint32_t* indexes = &((uint32_t*)temp)[offset];
        offset += radixPasses;
        uint32_t* globalHist = &((uint32_t*)temp)[offset];
        offset += radixPasses * RADIX;
        uint32_t* passHists = &((uint32_t*)temp)[offset];
        offset += binningPartitions * radixPasses * RADIX;
        uint32_t* fix = &((uint32_t*)temp)[offset];         //This will be reinterpretted as necessary

        //Clear
        cudaMemset(indexes, 0, (radixPasses * (RADIX + 1)) +
            (binningPartitions * radixPasses * RADIX) * sizeof(uint32_t));

        DispatchGlobalHistAndFix<V, BITS_TO_SORT>(
            segments,
            binOffsets,
            largeSegmentOffsets,
            sort,
            values,
            altA,
            altPayloadA,
            fix,
            globalHist,
            totalSegCount,
            totalSegLength,
            segsInCurBin,
            radixPasses,
            size);

        LargeScan<<<radixPasses, RADIX>>>(
            globalHist,
            passHists,
            binningPartitions);

        //There is probably a way to do this that is elegant and performant. . .
        //If you are reading this, suggestions welcome :^)
        switch (radixPasses)
        {
        case 5:
        {
            uint8_t* fixA = reinterpret_cast<uint8_t*>(fix);
            uint8_t* fixB = &reinterpret_cast<uint8_t*>(fix)[size];
            DigitBinningPass<
                uint32_t,
                V,
                uint8_t,
                k_warps,
                k_keysPerThread,
                0,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altA,
                altPayloadA,
                fixA,
                altB,
                altPayloadB,
                fixB,
                passHists,
                indexes,
                size);

            DigitBinningPass<
                uint32_t,
                V,
                uint8_t,
                k_warps,
                k_keysPerThread,
                8,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altB,
                altPayloadB,
                fixB,
                altA,
                altPayloadA,
                fixA,
                &passHists[binningPartitions * RADIX],
                &indexes[1],
                size);

            DigitBinningPass<
                uint32_t,
                V,
                uint8_t,
                k_warps,
                k_keysPerThread,
                16,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altA,
                altPayloadA,
                fixA,
                altB,
                altPayloadB,
                fixB,
                &passHists[binningPartitions * RADIX * 2],
                &indexes[2],
                size);

            DigitBinningPass<
                uint32_t,
                V,
                uint8_t,
                k_warps,
                k_keysPerThread,
                24,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altB,
                altPayloadB,
                fixB,
                altA,
                altPayloadA,
                fixA,
                &passHists[binningPartitions * RADIX * 3],
                &indexes[3],
                size);

            DigitBinningPass<
                uint8_t,
                V,
                uint32_t,
                k_warps,
                k_keysPerThread,
                0,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                fixA,
                altPayloadA,
                altA,
                fixB,
                altPayloadB,
                altB,
                &passHists[binningPartitions * RADIX * 4],
                &indexes[4],
                size);

            ScatterBack<V, uint8_t, 256 * 8, 8><<<dvrup<2048>(size), 256>>>(
                segments,
                largeSegmentOffsets,
                altB,
                altPayloadB,
                fixB,
                sort,
                values,
                size,
                totalSegLength);
            break;
        }
        case 6:
        {
            uint16_t* fixA = reinterpret_cast<uint16_t*>(fix);
            uint16_t* fixB = &reinterpret_cast<uint16_t*>(fix)[size];
            DigitBinningPass<
                uint32_t,
                V,
                uint16_t,
                k_warps,
                k_keysPerThread,
                0,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altA,
                altPayloadA,
                fixA,
                altB,
                altPayloadB,
                fixB,
                passHists,
                indexes,
                size);

            DigitBinningPass<
                uint32_t,
                V,
                uint16_t,
                k_warps,
                k_keysPerThread,
                8,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altB,
                altPayloadB,
                fixB,
                altA,
                altPayloadA,
                fixA,
                &passHists[binningPartitions * RADIX],
                &indexes[1],
                size);

            DigitBinningPass<
                uint32_t,
                V,
                uint16_t,
                k_warps,
                k_keysPerThread,
                16,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altA,
                altPayloadA,
                fixA,
                altB,
                altPayloadB,
                fixB,
                &passHists[binningPartitions * RADIX * 2],
                &indexes[2],
                size);

            DigitBinningPass<
                uint32_t,
                V,
                uint16_t,
                k_warps,
                k_keysPerThread,
                24,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                altB,
                altPayloadB,
                fixB,
                altA,
                altPayloadA,
                fixA,
                &passHists[binningPartitions * RADIX * 3],
                &indexes[3],
                size);

            DigitBinningPass<
                uint16_t,
                V,
                uint32_t,
                k_warps,
                k_keysPerThread,
                0,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                fixA,
                altPayloadA,
                altA,
                fixB,
                altPayloadB,
                altB,
                &passHists[binningPartitions * RADIX * 4],
                &indexes[4],
                size);

            DigitBinningPass<
                uint16_t,
                V,
                uint32_t,
                k_warps,
                k_keysPerThread,
                8,
                true>
            <<<binningPartitions, k_warps * LANE_COUNT>>>(
                fixB,
                altPayloadB,
                altB,
                fixA,
                altPayloadA,
                altA,
                &passHists[binningPartitions * RADIX * 5],
                &indexes[5],
                size);

            ScatterBack<V, uint16_t, 256 * 8, 8><<<dvrup<2048>(size), 256>>>(
                segments,
                largeSegmentOffsets,
                altA,
                altPayloadA,
                fixA,
                sort,
                values,
                size,
                totalSegLength);
            break;
        }
        case 7:
        {

            break;
        }
        case 8:
        {
            break;
        }
        default:
            break;
        }
        

        cudaFree(temp);

        cudaError_t cuda_err;
        cuda_err = cudaGetLastError();
        CUDA_CHECK(cuda_err, "Err1");
    }
}

#undef RADIX_LOG
#undef RADIX_MASK
#undef RADIX

#undef FLAG_MASK
#undef FLAG_INCLUSIVE
#undef FLAG_REDUCTION
#undef FLAG_NOT_READY