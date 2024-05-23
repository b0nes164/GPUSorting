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

namespace SplitSort 
{
    //w4_t32_kv32_cute32_bin
    template<uint32_t BITS_TO_SORT>
    __global__ void SortLe32(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* minBinSegCounts,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBins32<32, 128, 4, BITS_TO_SORT>(
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
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt32Le64(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<2, 64, 256, 4, 5, 6>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    //w1_t64_kv128_cute64_bMerge
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt64Le128(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<2, 64, 128, 1, 2, 6, 6, 7>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    //w1_t128_kv256_cute64_bMerge
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt128Le256(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<2, 64, 256, 1, 4, 6, 6, 8>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    //w1_t128_kv512_radix
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt256Le512(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<4, 4, 128, 512, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //w1_t128_kv1024_radix
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt512Le1024(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<4, 8, 256, 1024, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //w1_t256_kv2048_radix
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt1024Le2048(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<8, 8, 256, 2048, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //w1_t512_kv4096_radix
    template<uint32_t BITS_TO_SORT>
    __global__ void SortGt2048Le4096(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<16, 8, 256, 4096, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //onesweep here :)
}