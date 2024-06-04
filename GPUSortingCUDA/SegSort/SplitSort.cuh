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
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Utils.cuh"
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

    //w4_t32_kv128_cute32_wMerge
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
        SplitSortWarp<4, 128, 512, 4, 5, 7, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 4>);
    }

    //w1_t128_kv256_cute32_bMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt128Le256(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<2, 64, 256, 4, 5, 6, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    //w1_t128_kv512_cute64_bMerge
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void SortGt256Le512(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        if constexpr (BITS_TO_SORT > 24)
        {
            SplitSortBlock<4, 128, 512, 4, 6, 7, 9, false, K>(
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
            SplitSortRadix<4, 4, 128, 512, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                segments,
                binOffsets,
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }
    }

    //w1_t256_kv1024_cute64_bMerge
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
            SplitSortBlock<4, 128, 1024, 8, 6, 7, 10, false, K>(
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
            SplitSortRadix<8, 4, 128, 1024, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
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

    //global sort here aka onesweep :)

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
        cudaStream_t streams[k_segHistSize - 1];
        for (uint32_t i = 0; i < k_segHistSize - 1; ++i)
            cudaStreamCreate(&streams[i]);

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
            SplitSort::SortGt32Le64<BITS_TO_SORT><<<(segsInCurBin + 3) / 4, 128, 0, streams[1]>>>(
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
            SplitSort::SortGt64Le128<BITS_TO_SORT><<<(segsInCurBin + 3) / 4, 128, 0, streams[2]>>>(
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
            SplitSort::SortGt128Le256<BITS_TO_SORT><<<segsInCurBin, 128, 0, streams[3]>>>(
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
            SplitSort::SortGt256Le512<BITS_TO_SORT><<<segsInCurBin, 128, 0, streams[4]>>>(
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
            SplitSort::SortGt512Le1024<BITS_TO_SORT><<<segsInCurBin, 256, 0, streams[5]>>>(
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
            SplitSort::SortGt1024Le2048<BITS_TO_SORT><<<segsInCurBin, 256, 0, streams[6]>>>(
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
            SplitSort::SortGt2048Le4096<BITS_TO_SORT><<<segsInCurBin, 512, 0, streams[7]>>>(
                segments,
                binOffsets + segHist[7],
                sort,
                payloads,
                totalSegCount,
                totalSegLength);
        }

        for (uint32_t i = 0; i < k_segHistSize - 1; ++i)
            cudaStreamDestroy(streams[i]);
    }
}