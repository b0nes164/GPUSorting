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

namespace SplitSortInternal
{
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

    template<uint32_t RADIX_PASSES>
    __device__ __forceinline__ void Hist(
        const uint8_t* key8,
        uint32_t* s_warpsHist0,
        uint32_t* s_warpsHist1,
        uint32_t* s_warpsHist2,
        uint32_t* s_warpsHist3)
    {
        count<RADIX_PASSES>(
            key8,
            s_warpsHist0,
            s_warpsHist1,
            s_warpsHist2,
            s_warpsHist3);

        count<RADIX_PASSES>(
            &key8[4],
            s_warpsHist0,
            s_warpsHist1,
            s_warpsHist2,
            s_warpsHist3);

        count<RADIX_PASSES>(
            &key8[8],
            s_warpsHist0,
            s_warpsHist1,
            s_warpsHist2,
            s_warpsHist3);

        count<RADIX_PASSES>(
            &key8[12],
            s_warpsHist0,
            s_warpsHist1,
            s_warpsHist2,
            s_warpsHist3);
    }

    template<uint32_t RADIX>
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

    //Note fix is not vec striped, however this is not an
    //issue as the segId is a constant value
    template<class F>
    __device__ __forceinline__ void Fix(F* fix)
    {
        fix[threadIdx.x] = blockIdx.x;

        fix += blockDim.x;
        fix[threadIdx.x] = blockIdx.x;

        fix += blockDim.x;
        fix[threadIdx.x] = blockIdx.x;

        fix += blockDim.x;
        fix[threadIdx.x] = blockIdx.x;
    }

    //We already have the seglength, which is the digit count 
    //of segIds. So we add this directly to the global hist
    template<uint32_t RADIX, uint32_t FIX_BYTES>
    __device__ __forceinline__ void AddSegIdDigitCounts(
        uint32_t* globalHistogram,
        const uint32_t totalLocalLength)
    {
        uint32_t segId = blockIdx.x;
        uint8_t* segId8 = reinterpret_cast<uint8_t*>(segId);

        if (!threadIdx.x)
        {
            constexpr uint32_t RADIX_OFFSET_FIFTH = RADIX * 4;
            atomicAdd(&globalHistogram[segId8[0] + RADIX_OFFSET_FIFTH], totalLocalLength);

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

    //TODO keys must always be written to a seperate buffer
    template<
        class F,
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        uint32_t RADIX,
        uint32_t RADIX_PASSES,
        uint32_t PART_SIZE>
    __global__ void GlobalHistAndFix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* globalHistogram,
        F* fix,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
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

        //Advance pointer to segment start
        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;

        //clear
        for (uint32_t i = threadIdx.x; i < sMemSize; i += blockDim.x)
        {
            s_hist0[i] = 0;
            s_hist1[i] = 0;
            s_hist2[i] = 0;
            s_hist3[i] = 0;
        }
        __syncthreads();

        uint32_t part = 0;
        uint32_t partAlignedSize = totalLocalLength / PART_SIZE * PART_SIZE;
        constexpr uint32_t ADJUST_PART = PART_SIZE / 4;

        //If the number of passes is less than 4, we pack the segIds into the key
        //else, a seperate buffer holds the value
        if constexpr (RADIX_PASSES <= 4)
        {
            for (; part < partAlignedSize; part += PART_SIZE)
            {
                //Load
                uint4 keys[1] = { reinterpret_cast<uint4*>(sort)[threadIdx.x + part] };
                uint32_t* key32 = reinterpret_cast<uint32_t*>(keys);

                //Pack seg id
                key32[0] |= blockIdx.x << BITS_TO_SORT;
                key32[1] |= blockIdx.x << BITS_TO_SORT;
                key32[2] |= blockIdx.x << BITS_TO_SORT;
                key32[3] |= blockIdx.x << BITS_TO_SORT;

                //Hist
                Hist<RADIX_PASSES>(
                    reinterpret_cast<uint8_t*>(keys),
                    s_warpsHist0,
                    s_warpsHist1,
                    s_warpsHist2,
                    s_warpsHist3);

                //Write the "fixed" keys back
                reinterpret_cast<uint4*>(sort)[threadIdx.x + part] = keys[0];
            }

            //last part is not vectorized
            constexpr uint32_t ADJUST_PART = PART_SIZE / 4;
            for (; part < totalLocalLength; part += ADJUST_PART)
            {
                uint32_t key = sort[threadIdx.x + part];
                key |= blockIdx.x << BITS_TO_SORT;
                count<RADIX_PASSES>(
                    reinterpret_cast<uint8_t*>(key),
                    s_warpsHist0,
                    s_warpsHist1,
                    s_warpsHist2,
                    s_warpsHist3);
                sort[threadIdx.x + part] = key; //TODO <------------------------------------------------
            }
        }
        else
        {
            for (; part < partAlignedSize; part += PART_SIZE)
            {
                //Load
                uint4 keys[1] = { reinterpret_cast<uint4*>(sort)[threadIdx.x + part] };

                //Hist
                Hist<4>(
                    reinterpret_cast<uint8_t*>(keys),
                    s_warpsHist0,
                    s_warpsHist1,
                    s_warpsHist2,
                    s_warpsHist3);

                //Place the segIds into the fix buffer
                Fix(fix + part);
            }

            //last part is not vectorized
            for (; part < totalLocalLength; part += ADJUST_PART)
            {
                uint32_t key = sort[threadIdx.x + part];
                key |= blockIdx.x << BITS_TO_SORT;
                count<4>(
                    reinterpret_cast<uint8_t*>(key),
                    s_warpsHist0,
                    s_warpsHist1,
                    s_warpsHist2,
                    s_warpsHist3);
                //TODO <------------------------------------------------
                fix[threadIdx.x + part] = blockIdx.x;
            }
        }
        __syncthreads();

        ReduceAndAddDigitCounts<RADIX>(
            globalHistogram,
            s_hist0,
            s_hist1,
            s_hist2,
            s_hist3);

        //if the number of radix passes is greater than 4
        //then the segId's need to be histogrammed seperately.
        if constexpr (RADIX_PASSES > 4)
        {
            AddSegIdDigitCounts<RADIX, sizeof(F)>(
                globalHistogram,
                totalLocalLength);
        }
    }

    //We want: 
    //1) The number of radixPasses to be compile time visible
    //2) To vary the number of bytes used for the fix buffer, if any
    //so we wrap templates of the dispatch in a switch statement. Maybe there
    //is a better way to do this?
    template<uint32_t BITS_TO_SORT>
    __host__ void DispatchGlobalHistAndFix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* globalHistogram,
        uint32_t* fix,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segsInCurBin,
        const uint32_t radixPasses)
    {
        switch (radixPasses)
        {
        case 1:
            GlobalHistAndFix<uint32_t, 4, BITS_TO_SORT, 256, 1, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                fix,
                totalSegCount,
                totalSegLength);
            break;
        case 2:
            GlobalHistAndFix<uint32_t, 4, BITS_TO_SORT, 256, 2, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                fix,
                totalSegCount,
                totalSegLength);
            break;
        case 3:
            GlobalHistAndFix<uint32_t, 4, BITS_TO_SORT, 256, 3, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                fix,
                totalSegCount,
                totalSegLength);
            break;
        case 4:
            GlobalHistAndFix<uint32_t, 4, BITS_TO_SORT, 256, 4, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                fix,
                totalSegCount,
                totalSegLength);
            break;
        case 5:
            GlobalHistAndFix<uint8_t, 4, BITS_TO_SORT, 256, 5, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                reinterpret_cast<uint8_t*>(fix),
                totalSegCount,
                totalSegLength);
            break;
        case 6:
            GlobalHistAndFix<uint16_t, 4, BITS_TO_SORT, 256, 6, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                reinterpret_cast<uint16_t*>(fix),
                totalSegCount,
                totalSegLength);
            break;
        case 7:
            GlobalHistAndFix<uint32_t, 4, BITS_TO_SORT, 256, 7, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                fix,
                totalSegCount,
                totalSegLength);
            break;
        case 8:
            GlobalHistAndFix<uint32_t, 4, BITS_TO_SORT, 256, 8, 512><<<segsInCurBin, 128>>>(
                segments,
                binOffsets,
                sort,
                globalHistogram,
                fix,
                totalSegCount,
                totalSegLength);
            break;
        default:
            //TODO ASSERT ERROR or something
            break;
        }
    }

    template<class V, uint32_t BITS_TO_SORT>
    __host__ void SplitSortLarge(
        uint32_t* segments,
        uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segsInCurBin,
        const uint32_t size)            //TODO <--- read this back
    {
        const uint32_t k_radix = 256;
        const uint32_t k_radixLog = 8;
        const uint32_t k_partSize = 3840;
        const uint32_t segIdBits = findHighestBit(segsInCurBin);
        const uint32_t totalBits = segIdBits + BITS_TO_SORT;
        const uint32_t radixPasses = totalBits <= 32 ?
            dvrup<k_radixLog>(totalBits) : 4 + dvrup<k_radixLog>(segIdBits);
        const uint32_t binningPartitions = (size + k_partSize - 1) / k_partSize;
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
            (size * sizeof(V)) +                            //Alt payload
            (size * sizeof(uint32_t)) +                     //Alt keys
            (size * fixBytes * 2) +                         //Fix buffer, if necessary
            ((radixPasses +                                 //Chained scan atomic bump
            (radixPasses * k_radix) +                       //GlobalHistogram
            (binningPartitions * radixPasses * k_radix))    //PassHistograms
            * sizeof(uint32_t)));

        V* altPayload = (V*)temp;
        uint32_t offset = size * (sizeof(V) / sizeof(uint32_t));
        uint32_t* alt = &((uint32_t*)temp)[offset];
        offset += size;
        uint32_t* indexes = &((uint32_t*)temp)[offset];
        offset += radixPasses;
        uint32_t* globalHist = &((uint32_t*)temp)[offset];
        offset += radixPasses * k_radix;
        uint32_t* passHists = &((uint32_t*)temp)[offset];
        offset += binningPartitions * radixPasses * k_radix;
        uint32_t* fix = &((uint32_t*)temp)[offset];         //This will be reinterpretted as necessary

        //Clear
        cudaMemset(indexes, 0, (radixPasses * (k_radix + 1)) +
            (binningPartitions * radixPasses * k_radix) * sizeof(uint32_t));

        DispatchGlobalHistAndFix<BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            globalHist,
            fix,
            totalSegCount,
            totalSegLength,
            segsInCurBin,
            radixPasses);


        cudaFree(temp);
    }
}