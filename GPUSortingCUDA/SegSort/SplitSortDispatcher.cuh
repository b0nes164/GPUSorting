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
#include "SplitSortBinning.cuh"
#include "SplitSort.cuh"
#include "../UtilityKernels.cuh"

template<class K>
class SplitSortDispatcher
{
    const uint32_t k_nextFitSize = 2048;
    const uint32_t k_nextFitThreads = 64;
    const uint32_t k_segHistSize = 9;

    const uint32_t k_maxSize;
    const uint32_t k_maxSegments;

    uint32_t* m_sort;
    K* m_payloads;
    uint32_t* m_segments;
    uint32_t* m_minBinSegCounts;
    uint32_t* m_binOffsets;
    uint32_t* m_reduction;
    uint32_t* m_segHist;
    uint32_t* m_index;
    uint32_t* m_totalLength;
    uint32_t* m_errCount;

    int* a;
    int* b;

public:
    template<class K>
    SplitSortDispatcher(
        uint32_t maxSize,
        uint32_t maxSegments,
        K dummy) :
        k_maxSize(maxSize),
        k_maxSegments(maxSegments)
    {
        cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_payloads, k_maxSize * sizeof(K));
        cudaMalloc(&m_segments, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_minBinSegCounts, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_binOffsets, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_reduction, divRoundUp(k_maxSegments, k_nextFitSize) * sizeof(uint32_t));
        cudaMalloc(&m_segHist, k_segHistSize * sizeof(uint32_t));
        cudaMalloc(&m_index, sizeof(uint32_t));
        cudaMalloc(&m_totalLength, sizeof(uint32_t));
        cudaMalloc(&m_errCount, sizeof(uint32_t));
    }

    ~SplitSortDispatcher()
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

    void BatchTimeFixedSegmentLength(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        if (totalSegCount > k_maxSegments)
        {
            printf("Error seg count exceed max allocated memory. \n");
            return;
        }

        if (totalSegCount * segLength > k_maxSize)
        {
            printf("Error sort size exceeds max allocated memory. \n");
            return;
        }

        printf("Beginning Split Sort Pairs Fixed Seg Length Batch Timing: \n");
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            InitSegLengthsFixed<<<256,256>>>(m_segments, totalSegCount, segLength);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_sort, segLength, totalSegCount);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_payloads, segLength, totalSegCount);
            InitFixedSegLengthRandomValue<<<1024,64>>>(m_sort, m_payloads, segLength, totalSegCount, i + 10);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            DispatchSplitSortPairs<32>(totalSegCount, totalSegCount * segLength);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        totalTime /= 1000.0f;
        uint32_t size = totalSegCount * segLength;

        printf("\n");
        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit elements: %E pairs/sec\n\n", size, size / totalTime * batchCount);
    }

    template<uint32_t BITS_TO_SORT>
    void TestAllFixedSegmentLengths(uint32_t testsPerSegmentLength)
    {
        if (k_maxSize < (1 << 27))
        {
            printf("Error fixed segment length test requires 2^27 allocated sort memory. \n");
            return;
        }

        if (!testsPerSegmentLength)
        {
            printf("Error at least one test is required at each segment length. \n");
            return;
        }

        const uint32_t segCount = 1 << 13;
        printf("Beginning Split Sort Test All Fixed Segment Lengths 1 - 4096 \n");
        
        uint32_t testsPassed = 0;
        for (uint32_t segLength = 1; segLength <= 4096; ++segLength)
        {
            for (uint32_t i = 0; i < testsPerSegmentLength; ++i)
            {
                InitSegLengthsFixed<<<256,256>>>(m_segments, segCount, segLength);
                InitFixedSegLengthRandomValue<<<1024,64>>>(m_sort, m_payloads, segLength, segCount, i + 10);
                DispatchSplitSortPairs<BITS_TO_SORT>(segCount, segLength * segCount);
                if (ValidateSegSortFixedLength(segCount, segLength, false))
                    testsPassed++;
                else
                    printf("Test failed at fixed seg length: %u \n", segLength);
            }

            if ((segLength & 63) == 0)
                printf(". ");
        }

        const uint32_t testsExpected = 4096 * testsPerSegmentLength;
        if (testsPassed == testsExpected)
            printf("SPLIT SORT ALL FIXED SEG LENGTHS TESTS PASSED \n");
        else
            printf("SPLIT SORT FIXED SEG LENGTH TESTS FAILED. \n");
    }

private:
    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    template<uint32_t BITS_TO_SORT>
    void DispatchSplitSortPairs(uint32_t totalSegCount, uint32_t totalSegLength)
    {
        SplitSort::SplitSortPairs<BITS_TO_SORT>(
            m_segments,
            m_binOffsets,
            m_sort,
            m_payloads,
            m_segHist,
            m_index,
            m_reduction,
            m_minBinSegCounts,
            totalSegCount,
            totalSegLength);
    }

    bool ValidateSegSortFixedLength(uint32_t segCount, uint32_t segLength, bool shouldPrint)
    {
        uint32_t errCount[1];
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        ValidateFixLengthSegments<<<256,256>>>(m_sort, m_payloads, m_errCount, segLength, segCount);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (shouldPrint && errCount[0])
            Print<<<1,1>>>(m_sort, segCount * segLength);
        cudaDeviceSynchronize();
        return !errCount[0];
    }
};