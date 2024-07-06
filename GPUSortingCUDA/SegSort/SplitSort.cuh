/******************************************************************************
 * GPUSorting
 * SplitSort
 * Experimental Hybrid Radix-Merge based SegmentedSort
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 6/17/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"
#include "SplitSortVariants.cuh"
#include "SplitSortBinning.cuh"

namespace SplitSort 
{
    //w4_t32_kv32_cute32_bin
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortLe32(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* minBinSegCounts,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBins32<32, 128, 4, BITS_TO_SORT, K>(
            segments,
            binOffsets,
            minBinSegCounts,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin);
    };

    //w4_t32_kv64_cute64_wMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt32Le64(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<2, 64, 256, 4, 5, 6, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    //w2_t32_kv128_cute64_wMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt64Le128(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<4, 128, 256, 2, 6, 7, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    //w1_t64_kv256_cute64_bMerge
    //w1_t64_kv256_cute128_bMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt128Le256(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        if constexpr (BITS_TO_SORT > 24)
        {
            SplitSortBlock<4, 128, 256, 2, 6, 7, 8, false, K>(
                segments,
                binOffsets,
                sort,
                payloads,
                totalSegCount,
                totalSegLength,
                CuteSort64<BITS_TO_SORT, 4>);
        }
        else
        {
            SplitSortBlock<4, 128, 256, 2, 7, 7, 8, false, K>(
                segments,
                binOffsets,
                sort,
                payloads,
                totalSegCount,
                totalSegLength,
                CuteSort128<BITS_TO_SORT, 4>);
        }
    }

    //w1_t128_kv512_cute128_bMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt256Le512(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<4, 128, 512, 4, 7, 7, 9, false, K>(
        segments,
        binOffsets,
        sort,
        payloads,
        totalSegCount,
        totalSegLength,
        CuteSort128<BITS_TO_SORT, 4>);
    }

    //w1_t256_kv1024_cute128_bMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt512Le1024(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        if constexpr (BITS_TO_SORT > 24)
        {
            SplitSortBlock<4, 128, 1024, 8, 7, 7, 10, false, K>(
                segments,
                binOffsets,
                sort,
                payloads,
                totalSegCount,
                totalSegLength,
                CuteSort128<BITS_TO_SORT, 4>);
        }
        else
        {
            SplitSortRadix<4, 8, 256, 1024, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                segments,
                binOffsets,
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }
    }

    //w1_t256_kv2048_radix
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt1024Le2048(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<8, 8, 256, 2048, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //w1_t512_kv4096_radix
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt2048Le4096(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<16, 8, 256, 4096, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //TODO
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt4096Le6144(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        __shared__ uint32_t s_mem[6144 * 2];
        uint32_t totalLocalLength;
        GetSegmentInfoRadixFine(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            totalLocalLength);

        if (totalLocalLength <= 4608)
        {
            SplitSortRadixFine<16, 9, 288, 4608, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 4608 && totalLocalLength <= 5120)
        {
            SplitSortRadixFine<16, 10, 320, 5120, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 5120 && totalLocalLength <= 5632)
        {
            SplitSortRadixFine<16, 11, 352, 5632, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 5632 && totalLocalLength <= 6144)
        {
            SplitSortRadixFine<16, 12, 384, 6144, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }
        
        //Kern too long, have to split

        /*
        if (totalLocalLength > 6144 && totalLocalLength <= 6656)
        {
            SplitSortRadixFine<16, 13, 416, 6656, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[8192], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 6656 && totalLocalLength <= 7168)
        {
            SplitSortRadixFine<16, 14, 448, 7168, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[8192], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 7168 && totalLocalLength <= 7680)
        {
            SplitSortRadixFine<16, 15, 480, 7680, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[8192], //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 7680)
        {
            SplitSortRadixFine<16, 16, 512, 8192, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[8192],  //RADIX * WARPS or PART_SIZE
                sort,
                payloads,
                totalLocalLength);
        }
        */
    }

    //FixSort

    template<uint32_t BITS_TO_SORT, class K>
    __host__ void SplitSortPairs(
        uint32_t* segments,
        uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        uint32_t* tempSegHist,
        uint32_t* tempIndex,
        uint32_t* tempBinReductions,
        uint32_t* tempMinBinSegCounts,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        const uint32_t k_segHistSize = 9;
        const uint32_t k_nextFitSize = 2048;

        uint32_t segHist[k_segHistSize];
        //cudaStream_t stream[9 - 1];
        //for(uint32_t i = 0; i < k_segHistSize - 1; ++i)
        //    cudaStreamCreate(&stream[i]);

        const uint32_t binPackPartitions = (totalSegCount + k_nextFitSize - 1) / k_nextFitSize;
        cudaMemset(tempIndex, 0, sizeof(uint32_t));
        cudaMemset(tempSegHist, 0, k_segHistSize * sizeof(uint32_t));
        cudaMemset(tempBinReductions, 0, binPackPartitions * sizeof(uint32_t));
        cudaDeviceSynchronize();

        SplitSortBinning::NextFitBinPacking<<<binPackPartitions, 64>>>(
            segments,
            tempSegHist,
            tempMinBinSegCounts,
            binOffsets,
            tempIndex,
            tempBinReductions,
            totalSegCount,
            totalSegLength);

        SplitSortBinning::Scan<<<1, 32>>>(tempSegHist);

        cudaMemcpyAsync(segHist, tempSegHist, k_segHistSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        SplitSortBinning::Bin<<<(totalSegCount + 255) / 256, 256>>>(
            segments,
            tempSegHist,
            binOffsets,
            totalSegCount,
            totalSegLength);

        uint32_t segsInCurBin = segHist[1] - segHist[0];
        if (segsInCurBin)
        {
            SplitSort::SortLe32<BITS_TO_SORT><<<(segsInCurBin + 3) / 4, 128>>>(
                segments,
                binOffsets,
                tempMinBinSegCounts,
                sort,
                payloads,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }
        
        segsInCurBin = segHist[2] - segHist[1];
        if (segsInCurBin)
        {
            SplitSort::SortGt32Le64<BITS_TO_SORT><<<(segsInCurBin + 3) / 4, 128>>>(
                segments,
                binOffsets + segHist[1],
                sort,
                payloads,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segHist[3] - segHist[2];
        if (segsInCurBin)
        {
            SplitSort::SortGt64Le128<BITS_TO_SORT><<<(segsInCurBin + 1) / 2, 64>>>(
                segments,
                binOffsets + segHist[2],
                sort,
                payloads,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segHist[4] - segHist[3];
        if (segsInCurBin)
        {
            SplitSort::SortGt128Le256<BITS_TO_SORT><<<segsInCurBin, 64>>>(
                segments,
                binOffsets + segHist[3],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[5] - segHist[4];
        if (segsInCurBin)
        {
            SplitSort::SortGt256Le512<BITS_TO_SORT><<<segsInCurBin, 128>>>(
                segments,
                binOffsets + segHist[4],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[6] - segHist[5];
        if (segsInCurBin)
        {
            constexpr uint32_t threads = BITS_TO_SORT > 24 ? 256 : 128;
            SplitSort::SortGt512Le1024<BITS_TO_SORT><<<segsInCurBin, threads>>>(
                segments,
                binOffsets + segHist[5],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[7] - segHist[6];
        if (segsInCurBin)
        {
            SplitSort::SortGt1024Le2048<BITS_TO_SORT><<<segsInCurBin, 256>>>(
                segments,
                binOffsets + segHist[6],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[8] - segHist[7];
        if (segsInCurBin)
        {
            SplitSort::SortGt2048Le4096<BITS_TO_SORT><<<segsInCurBin, 512>>>(
                segments,
                binOffsets + segHist[7],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = totalSegCount - segHist[8];
        if (segsInCurBin)
        {
            /*dim3 grids(512, 1, 1);
            cudaFuncSetAttribute(SortGt4096Le8192<BITS_TO_SORT, K>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            SplitSort::SortGt4096Le8192<BITS_TO_SORT, K><<<segsInCurBin, grids, 65536, 0>>>(
                segments,
                binOffsets + segHist[8],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);*/

            SplitSort::SortGt4096Le6144<BITS_TO_SORT, K><<<segsInCurBin, 512>>>(
                segments,
                binOffsets + segHist[8],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        //for(uint32_t i = 0; i < k_segHistSize - 1; ++i)
        //    cudaStreamDestroy(stream[i]);
    }
}