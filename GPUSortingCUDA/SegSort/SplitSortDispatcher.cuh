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
#include "cub/device/device_scan.cuh"
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
        cudaMalloc(&m_totalLength, 3 * sizeof(uint32_t));
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

    void BatchTimeRandomSegmentLength(
        uint32_t batchCount,
        uint32_t size,
        uint32_t maxSegLength)
    {
        if (size > k_maxSize || size > k_maxSegments )
        {
            printf("Error, allocate more memory :) \n");
            return;
        }

        printf("Beginning Split Sort Pairs Random Seg Length Batch Timing: \n");
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        uint64_t totalSize = 0;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            //Init
                uint32_t segInfo[3];
                cudaMemset(m_totalLength, 0, 3 * sizeof(uint32_t));
                cudaDeviceSynchronize();
                InitSegLengthsRandom<<<4096,64>>>(m_segments, m_totalLength, i + 10, size, maxSegLength);
                cudaDeviceSynchronize();
                cudaMemcpy(&segInfo, m_totalLength, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                void* d_temp_storage = NULL;
                size_t  temp_storage_bytes = 0;
                cub::DeviceScan::ExclusiveSum(
                    d_temp_storage, temp_storage_bytes,
                    m_segments, m_segments, segInfo[1]);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceScan::ExclusiveSum(
                    d_temp_storage, temp_storage_bytes,
                    m_segments, m_segments, segInfo[1]);
                cudaDeviceSynchronize();
                cudaFree(d_temp_storage);
                InitRandomSegLengthRandomValue<<<4096,64>>>(m_sort, m_payloads, m_segments, segInfo[1], segInfo[0], i + 10);
                //InitRandomSegLengthUniqueValue<<<4096,64>>>(m_sort, m_payloads, m_segments, segInfo[1], segInfo[0], i + 10);

                cudaDeviceSynchronize();
                cudaEventRecord(start);
                DispatchSplitSortPairs<32>(segInfo[1], segInfo[0]);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaDeviceSynchronize();

                float millis;
                cudaEventElapsedTime(&millis, start, stop);
                if (i)
                {
                    totalTime += millis;
                    totalSize += segInfo[0];
                }
                    
                if ((i & 15) == 0)
                    printf(". ");
        }

        totalTime /= 1000.0f;
        double tSize = totalSize;
        tSize /= (double)batchCount;
        printf("\n");
        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit elements: %E pairs/sec\n\n", (uint32_t)tSize, tSize / totalTime * batchCount);
    }

    //Test random segment lengths, with maximums at powers of two between 1 and 4096
    template<uint32_t BITS_TO_SORT>
    void TestAllRandomSegmentLengths(uint32_t testsPerSegmentLength, bool shouldPrintSegInfo)
    {
        if (!testsPerSegmentLength)
        {
            printf("Error at least one test is required at each segment length. \n");
            return;
        }

        if (k_maxSize < (1 << 21) || k_maxSegments < (1 << 21))
        {
            printf("Error, allocate more memory :) \n");
            return;
        }

        printf("Beginning Split Sort Test All Random Segment Lengths \n");
        uint32_t testsPassed = 0;
        for (uint32_t maxSegLength = 1; maxSegLength <= 4096; maxSegLength <<= 1)
        {
            for (uint32_t i = 0; i < testsPerSegmentLength; ++i)
            {
                //Init
                uint32_t segInfo[3];
                cudaMemset(m_totalLength, 0, 3 * sizeof(uint32_t));
                cudaDeviceSynchronize();
                InitSegLengthsRandom<<<4096,64>>>(m_segments, m_totalLength, i + 10, 1 << 21, maxSegLength);
                cudaDeviceSynchronize();
                cudaMemcpy(&segInfo, m_totalLength, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                void* d_temp_storage = NULL;
                size_t  temp_storage_bytes = 0;
                cub::DeviceScan::ExclusiveSum(
                    d_temp_storage, temp_storage_bytes,
                    m_segments, m_segments, segInfo[1]);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceScan::ExclusiveSum(
                    d_temp_storage, temp_storage_bytes,
                    m_segments, m_segments, segInfo[1]);
                cudaDeviceSynchronize();
                cudaFree(d_temp_storage);
                InitRandomSegLengthRandomValue <<<4096,64>>>(m_sort, m_payloads, m_segments, segInfo[1], segInfo[0], i + 10);
                //InitRandomSegLengthUniqueValue<<<4096,64>>>(m_sort, m_payloads, m_segments, segInfo[1], segInfo[0], i + 10);
                if (shouldPrintSegInfo)
                {
                    printf("\n Beginning test: Total Segment Length: %u. Total Segment Count: %u. Max Segment Length %u\n",
                        segInfo[0], segInfo[1], maxSegLength);
                }
                else
                {
                    if ((i & 3) == 0)
                        printf(". ");
                }

                DispatchSplitSortPairs<BITS_TO_SORT>(segInfo[1], segInfo[0]);
                if (ValidateSegSortRandomLength(segInfo[1], segInfo[0], false))
                    testsPassed++;
                else
                    printf("Test failed at max seg length: %u \n", maxSegLength);
            }
        }

        const uint32_t testsExpected = 13 * testsPerSegmentLength;
        if (testsPassed == testsExpected)
            printf("\nSPLIT SORT ALL RANDOM SEG LENGTHS TESTS PASSED \n");
        else
            printf("\nSPLIT SORT FIXED RANDOM LENGTH TESTS FAILED. \n");
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
        ValidateFixLengthSegments<<<4096,64>>>(m_sort, m_payloads, m_errCount, segLength, segCount);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (shouldPrint && errCount[0])
            Print<<<1,1>>>(m_sort, segCount * segLength);
        cudaDeviceSynchronize();
        return !errCount[0];
    }

    bool ValidateSegSortRandomLength(uint32_t segCount, uint32_t totalSegLength, bool shouldPrint)
    {
        uint32_t errCount[1];
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        ValidateRandomLengthSegments<<<4096,64>>>(m_sort, m_payloads, m_segments, m_errCount, totalSegLength, segCount);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (shouldPrint && errCount[0])
            Print<<<1,1>>>(m_sort, totalSegLength);
        cudaDeviceSynchronize();
        return !errCount[0];
    }
};