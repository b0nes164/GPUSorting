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
#include "SplitSortVariants.cuh"
#include "SplitSortBinning.cuh"
#include "../UtilityKernels.cuh"

class SplitSortVariantTester
{
    const uint32_t k_nextFitSize = 2048;
    const uint32_t k_nextFitThreads = 64;
    const uint32_t k_segHistSize = 9;

    const uint32_t k_maxSize;
    const uint32_t k_maxSegments;

    uint32_t* m_sort;
    uint32_t* m_payloads;
    uint32_t* m_segments;
    uint32_t* m_minBinSegCounts;
    uint32_t* m_binOffsets;
    uint32_t* m_reduction;
    uint32_t* m_segHist;
    uint32_t* m_index;
    uint32_t* m_totalLength;
    uint32_t* m_errCount;

public:
    SplitSortVariantTester(
        uint32_t maxSize,
        uint32_t maxSegments) :
        k_maxSize(maxSize),
        k_maxSegments(maxSegments)
    {
        cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_payloads, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_segments, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_minBinSegCounts, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_binOffsets, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_reduction, divRoundUp(k_maxSegments, k_nextFitSize) * sizeof(uint32_t));
        cudaMalloc(&m_segHist, k_segHistSize * sizeof(uint32_t));
        cudaMalloc(&m_index, sizeof(uint32_t));
        cudaMalloc(&m_totalLength, sizeof(uint32_t));
        cudaMalloc(&m_errCount, sizeof(uint32_t));
    }

    ~SplitSortVariantTester()
    {
        cudaFree(m_sort);
        cudaFree(m_payloads);
        cudaFree(m_segments);
        cudaFree(m_minBinSegCounts);
        cudaFree(m_binOffsets);
        cudaFree(m_reduction);
        cudaFree(m_segHist);
        cudaFree(m_index);
        cudaFree(m_totalLength);
        cudaFree(m_errCount);
    }

    //for testing purposes only
    void Dispatch()
    {
        const uint32_t totalSegCount = 1;
        const uint32_t segLength = 160;

        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 256;

        uint32_t segHist[9];
        InitSegLengthsFixed<<<256,256>>>(m_segments, totalSegCount, segLength);
        InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_sort, segLength, totalSegCount);
        InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_payloads, segLength, totalSegCount);
        DispatchBinning(totalSegCount, totalSegCount * segLength, segHist);
        cudaDeviceSynchronize();

        SplitSortVariants::t64_kv256_cute32_bMerge<
            warpGroups * warpsPerWarpGroup,
            32><<<segHist[getSegHistIndex(kvProcessed)] / warpGroups, 32 * warpGroups * warpsPerWarpGroup>>>(
                m_segments,
                m_binOffsets,
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegCount * segLength);

        Print<<<1,1>>>(m_sort, totalSegCount * segLength);
        if (ValidateSegSortFixedLength(segLength, totalSegCount))
        {
            printf("Passed\n");
        }
        else
        {
            printf("Failed\n");
        }
    }

#pragma region SortTests
    void BatchTime_w1_t32_kv32_cute32_bin(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 1;

        BatchTimeBinSortFixedSegmentLength(
            "1 warp groups, 1 warp per warp group, 32 kv, cute32, bin",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            &SplitSortVariants::t32_kv32_cute32_bin<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv32_cute32_bin(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;

        BatchTimeBinSortFixedSegmentLength(
            "2 warp groups, 1 warp per warp group, 32 kv, cute32, bin",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            &SplitSortVariants::t32_kv32_cute32_bin<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w4_t32_kv32_cute32_bin(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 4;
        const uint32_t warpsPerWarpGroup = 1;

        BatchTimeBinSortFixedSegmentLength(
            "4 warp groups, 1 warp per warp group, 32 kv, cute32, bin",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            &SplitSortVariants::t32_kv32_cute32_bin<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t32_kv64_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 64;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 1 warp per warp group, 64 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv64_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv64_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 64;

        BatchTimeSortFixedSegmentLength(
            "2 warp groups, 1 warp per warp group, 64 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv64_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w4_t32_kv64_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 4;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 64;

        BatchTimeSortFixedSegmentLength(
            "4 warp groups, 1 warp per warp group, 64 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv64_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t32_kv64_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 64;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 1 warp per warp group, 64 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv64_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv64_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 64;

        BatchTimeSortFixedSegmentLength(
            "2 warp groups, 1 warp per warp group, 64 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv64_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w4_t32_kv64_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 4;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 64;

        BatchTimeSortFixedSegmentLength(
            "4 warp groups, 1 warp per warp group, 64 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv64_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t32_kv128_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 1 warp per warp group, 128 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv128_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv128_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "2 warp groups, 1 warp per warp group, 128 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv128_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w4_t32_kv128_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 4;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "4 warp groups, 1 warp per warp group, 128 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv128_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t32_kv128_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 1 warp per warp group, 128 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv128_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv128_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "2 warp groups, 1 warp per warp group, 128 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv128_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w4_t32_kv128_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 4;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "4 warp groups, 1 warp per warp group, 128 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv128_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t64_kv128_cute32_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 128 kv, cute32, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv128_cute32_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t64_kv128_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 128 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv128_cute64_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv256_cute32_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 256;

        BatchTimeSortFixedSegmentLength(
            "2 warp groups, 1 warps per warp group, 256 kv, cute32, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv256_cute32_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w2_t32_kv256_cute64_wMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 2;
        const uint32_t warpsPerWarpGroup = 1;
        const uint32_t kvProcessed = 256;

        BatchTimeSortFixedSegmentLength(
            "2 warp groups, 1 warps per warp group, 256 kv, cute64, warp merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t32_kv256_cute64_wMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t64_kv256_cute32_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 256;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 256 kv, cute32, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv256_cute32_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t64_kv256_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 256;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 256 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv256_cute64_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t64_kv512_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 512;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 512 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv512_cute64_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t128_kv512_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 4;
        const uint32_t kvProcessed = 512;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 4 warps per warp group, 512 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t128_kv512_cute64_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t256_kv512_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 8;
        const uint32_t kvProcessed = 512;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 8 warps per warp group, 512 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t256_kv512_cute64_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t128_kv1024_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 4;
        const uint32_t kvProcessed = 1024;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 4 warps per warp group, 1024 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t128_kv1024_cute64_bMerge<warpGroups* warpsPerWarpGroup, 32>);
    }

    void BatchTime_w1_t256_kv1024_cute64_bMerge(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 8;
        const uint32_t kvProcessed = 1024;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 8 warps per warp group, 1024 kv, cute64, block merge",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t256_kv1024_cute64_bMerge<warpGroups * warpsPerWarpGroup, 32>);
    }

    //RADIX SORTS
    void BatchTime_w1_t64_kv128_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 128;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 128 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv128_radix<32>);
    }

    void BatchTime_w1_t64_kv256_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 256;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 256 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv256_radix<32>);
    }

    void BatchTime_w1_t64_kv512_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 2;
        const uint32_t kvProcessed = 512;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 2 warps per warp group, 512 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t64_kv512_radix<32>);
    }

    void BatchTime_w1_t128_kv512_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 4;
        const uint32_t kvProcessed = 512;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 4 warps per warp group, 512 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t128_kv512_radix<32>);
    }

    void BatchTime_w1_t128_kv1024_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 4;
        const uint32_t kvProcessed = 1024;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 4 warps per warp group, 1024 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t128_kv1024_radix<32>);
    }

    void BatchTime_w1_t256_kv1024_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 8;
        const uint32_t kvProcessed = 1024;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 8 warps per warp group, 1024 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t256_kv1024_radix<32>);
    }

    void BatchTime_w1_t256_kv2048_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 8;
        const uint32_t kvProcessed = 2048;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 8 warps per warp group, 2048 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t256_kv2048_radix<32>);
    }

    void BatchTime_w1_t512_kv2048_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 16;
        const uint32_t kvProcessed = 2048;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 16 warps per warp group, 2048 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t512_kv2048_radix<32>);
    }

    void BatchTime_w1_t512_kv4096_radix(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        const uint32_t warpGroups = 1;
        const uint32_t warpsPerWarpGroup = 16;
        const uint32_t kvProcessed = 4096;

        BatchTimeSortFixedSegmentLength(
            "1 warp groups, 16 warps per warp group, 4096 kv, radix",
            batchCount,
            totalSegCount,
            segLength,
            warpGroups,
            getDispatchThreads(warpGroups, warpsPerWarpGroup),
            getSegHistIndex(kvProcessed),
            &SplitSortVariants::t512_kv4096_radix<32>);
    }
#pragma endregion SortTests

#pragma region BinningTests
    void BatchTimeBinningFixedSegLength(
        uint32_t batchCount,
        uint32_t segLength,
        uint32_t segCount)
    {

        cudaEvent_t start;
        cudaEvent_t stop;
        if (!BatchTimeSetup("Binning Fixed Seg Length", start, stop, segCount, segLength))
            return;

        uint32_t segHist[9];
        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            InitSegLengthsFixed <<<256, 256>>> (m_segments, segCount, segLength);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            DispatchBinning(segCount, segLength, segHist);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        //TODO add test
        PrintResults(batchCount + 1, segCount, batchCount, totalTime, "segs");
    }

    void BatchTimeBinningRandomSegLength(
        uint32_t batchCount,
        uint32_t maxSegLength,
        uint32_t segCount)
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        if (!BatchTimeSetup("Binning Random Seg Length", start, stop, segCount, 0))
            return;

        uint32_t segHist[9];
        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            cudaMemset(m_totalLength, 0, sizeof(uint32_t));
            cudaDeviceSynchronize();

            InitSegLengthsRandom<<<256,256>>>(
                m_segments,
                m_totalLength,
                ENTROPY_PRESET_1,
                i + 10,
                segCount,
                maxSegLength);
            cudaDeviceSynchronize();

            uint32_t totalLength[1];
            cudaMemcpy(&totalLength, m_totalLength, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            DispatchBinning(segCount, totalLength[0], segHist);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        //TODO add test
        PrintResults(batchCount + 1, segCount, batchCount, totalTime, "segs");
    }
#pragma endregion BinningTests

private:
    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    static inline uint32_t getSegHistIndex(uint32_t kvProcessed)
    {
        return __popcnt(kvProcessed - 1) - 5;
    }

    static inline uint32_t getDispatchThreads(uint32_t warpGroups, uint32_t warpsPerWarpGroup)
    {
        return 32 * warpGroups * warpsPerWarpGroup;  //Assumes 32 threads per warp
    }

    void DispatchBinning(uint32_t totalSegCount, uint32_t totalSegLength, uint32_t* segHist)
    {
        const uint32_t partitions = divRoundUp(totalSegCount, k_nextFitSize);
        cudaMemset(m_index, 0, sizeof(uint32_t));
        cudaMemset(m_segHist, 0, k_segHistSize * sizeof(uint32_t));
        cudaMemset(m_reduction, 0, partitions * sizeof(uint32_t));
        cudaDeviceSynchronize();

        SplitSortBinning::NextFitBinPacking<<<partitions, k_nextFitThreads>>>(
            m_segments,
            m_segHist,
            m_minBinSegCounts,
            m_binOffsets,
            m_index,
            m_reduction,
            totalSegCount,
            totalSegLength);
        cudaDeviceSynchronize();

        cudaMemcpy(segHist, m_segHist, k_segHistSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        SplitSortBinning::Scan<<<1,32>>>(m_segHist);

        SplitSortBinning::Bin<<<divRoundUp(totalSegCount,256), 256>>>(
            m_segments,
            m_segHist,
            m_binOffsets,
            totalSegCount,
            totalSegLength);
    }

    bool BatchTimeSetup(
        const char* testName,
        cudaEvent_t& start,
        cudaEvent_t& stop,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        if (totalSegCount > k_maxSegments)
        {
            printf("Error seg count exceed max allocated memory. \n");
            return false;
        }

        if (totalSegCount * segLength > k_maxSize)
        {
            printf("Error sort size exceeds max allocated memory. \n");
            return false;
        }

        printf("Beginning ");
        printf(testName);
        printf(" Batch Time \n");
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        return true;
    }

    static inline void PrintResults(
        uint32_t testsPassed,
        uint32_t size,
        uint32_t batchCount,
        float totalTime,
        const char* unit)
    {
        if (testsPassed == batchCount + 1)
            printf("\nValidation tests passed.");
        else
            printf("\nError validation %u tests failed.", batchCount + 1 - testsPassed);

        printf("\n");
        totalTime /= 1000.0f;
        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit ", size);
        printf(unit);
        printf(" %E ", size / totalTime * batchCount);
        printf(unit);
        printf("/sec\n\n");
    }

    //Time the sorting of segLengths <= 32
    void BatchTimeBinSortFixedSegmentLength(
        const char* sortName,
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength,
        uint32_t warpGroups,
        uint32_t sortThreads,
        void (*Sort)(
            const uint32_t*,
            const uint32_t*,
            const uint32_t*,
            uint32_t*,
            uint32_t*,
            const uint32_t,
            const uint32_t))
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        if (!BatchTimeSetup(sortName, start, stop, totalSegCount, segLength))
            return;

        uint32_t segHist[9];
        float totalTime = 0.0f;
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            InitSegLengthsFixed<<<256,256>>>(m_segments, totalSegCount, segLength);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_sort, segLength, totalSegCount);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_payloads, segLength, totalSegCount);
            InitFixedSegLengthRandomValue<<<1024,64>>>(m_sort, segLength, totalSegCount, i + 10);
            InitFixedSegLengthRandomValue<<<1024,64>>>(m_payloads, segLength, totalSegCount, i + 10);
            DispatchBinning(totalSegCount, totalSegCount * segLength, segHist);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            (*Sort)<<<segHist[0] / warpGroups, sortThreads>>>(
                m_segments,
                m_binOffsets,
                m_minBinSegCounts,
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegCount * segLength);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            if(ValidateSegSortFixedLength(segLength, totalSegCount))
                testsPassed++;

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        PrintResults(testsPassed, totalSegCount * segLength, batchCount, totalTime, "keys");
    }

    //Time the sorting of seg lengths > 32
    void BatchTimeSortFixedSegmentLength(
        const char* sortName,
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength,
        uint32_t warpGroups,
        uint32_t sortThreads,
        uint32_t segHistOffset,
        void (*Sort)(
            const uint32_t*,
            const uint32_t*,
            uint32_t*,
            uint32_t*,
            const uint32_t,
            const uint32_t))
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        if (!BatchTimeSetup(sortName, start, stop, totalSegCount, segLength))
            return;

        uint32_t segHist[9];
        float totalTime = 0.0f;
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            InitSegLengthsFixed<<<256,256>>>(m_segments, totalSegCount, segLength);
            //InitFixedSegLengthDescendingValue<<<1024,64>>>(m_sort, segLength, totalSegCount);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_payloads, segLength, totalSegCount);
            InitFixedSegLengthRandomValue<<<1024,64>>>(m_sort, segLength, totalSegCount, i + 10);
            InitFixedSegLengthRandomValue<<<1024,64>>>(m_payloads, segLength, totalSegCount, i + 10);
            DispatchBinning(totalSegCount, totalSegCount * segLength, segHist);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            (*Sort)<<<segHist[segHistOffset] / warpGroups, sortThreads>>>(
                m_segments,
                m_binOffsets,
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegCount * segLength);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            if (ValidateSegSortFixedLength(segLength, totalSegCount))
                testsPassed++;

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        PrintResults(testsPassed, totalSegCount * segLength, batchCount, totalTime, "keys");
    };

    bool ValidateSegSortFixedLength(uint32_t segLength, uint32_t segCount)
    {
        uint32_t errCount[1];
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        ValidateFixLengthSegments<<<256, 256>>>(m_sort, m_errCount, segLength, segCount);
        ValidateFixLengthSegments<<<256, 256>>>(m_payloads, m_errCount, segLength, segCount);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return !errCount[0];
    }
};