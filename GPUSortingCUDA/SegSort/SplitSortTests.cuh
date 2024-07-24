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
#include "cub/device/device_scan.cuh"
#include "cub/device/device_segmented_sort.cuh"
#include "SplitSort/SplitSort.cuh"
#include "../UtilityKernels.cuh"

#define SEG_INFO_SIZE 15

template<class K>
class SplitSortTests
{
    const uint32_t k_maxTotalLength;
    const uint32_t k_maxTotalSegCount;

    uint32_t* m_sort;
    K* m_payloads;
    uint32_t* m_segments;
    uint32_t* m_segInitInfo;
    uint32_t* m_segInfoValidate;
    uint32_t* m_errCount;

    void* m_tempMem;

public:
    template<class K>
    SplitSortTests(
        uint32_t maxTotalLength,
        uint32_t maxTotalSegCount,
        K dummy) :
        k_maxTotalLength(maxTotalLength),
        k_maxTotalSegCount(maxTotalSegCount)
    {
        cudaMalloc(&m_sort, k_maxTotalLength * sizeof(uint32_t));
        cudaMalloc(&m_payloads, k_maxTotalLength * sizeof(K));
        cudaMalloc(&m_segments, k_maxTotalSegCount * sizeof(uint32_t));
        cudaMalloc(&m_segInitInfo, 3 * sizeof(uint32_t));
        cudaMalloc(&m_segInfoValidate, SEG_INFO_SIZE * sizeof(uint32_t));
        cudaMalloc(&m_errCount, sizeof(uint32_t));

        SplitSortAllocateTempMemory(k_maxTotalLength, k_maxTotalSegCount, m_tempMem);
    }

    ~SplitSortTests()
    {
        cudaFree(m_sort);
        cudaFree(m_payloads);
        cudaFree(m_segments);
        cudaFree(m_segInitInfo);
        cudaFree(m_segInfoValidate);
        cudaFree(m_errCount);

        SplitSortFreeTempMemory(m_tempMem);
    }

    //TODO reexamine this test to ensure nothing broke
    void TestBinningRandomSegLength(
        const uint32_t testCount,
        const uint32_t maxSegLength,
        const uint32_t totalSegLength,
        const bool verbose)
    {
        if (k_maxTotalLength < (1 << 27))
        {
            printf("Error TestBinningRandomSegLength requires at least 1 << 27 allocated sort memory. \n");
            return;
        }

        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i < testCount; ++i)
        {
            //Init
            uint32_t segInitInfo[2];
            DispatchInitSegmentsRandomLengthRandomValue(segInitInfo, maxSegLength, totalSegLength, i + 10);
            cudaDeviceSynchronize();

            uint32_t segInfo[SEG_INFO_SIZE];
            const uint32_t nextFitPartitions = SplitSortInternal::GetNextFitPartitions(segInitInfo[1]);
            SplitSortInternal::SplitSortBinning(
                m_segments,
                SplitSortInternal::GetBinOffsetsPointer(m_tempMem, nextFitPartitions, segInitInfo[1]),
                SplitSortInternal::GetPackedSegCountsPointer(m_tempMem, nextFitPartitions),
                m_tempMem,
                segInfo,
                segInitInfo[1],
                segInitInfo[0],
                nextFitPartitions);
            cudaDeviceSynchronize();
            
            cudaMemcpy(m_segInfoValidate, segInfo, SEG_INFO_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            bool passed = ValidateBinning(segInitInfo[1], segInitInfo[0], segInfo[0], false); //Enable for super verbose
            if (passed)
                testsPassed++;

            if (verbose)
            {
                printf("Test %u: SegCount: %u TotalSegLength: %u \n", i, segInitInfo[1], segInitInfo[0]);
                if (passed)
                    printf("Test passed.\n");
                else
                    printf("Test failed.\n");
            }
            else
            {
                if ((i & 15) == 0)
                    printf(". ");
            }
        }

        if (testsPassed == testCount)
            printf("\nSPLIT SORT BINNING ALL TESTS PASSED\n");
        else
            printf("\nSPLIT SORT BINNING FAILED %u / %u \n", testsPassed, testCount);
    }

    template<uint32_t BITS_TO_SORT>
    void TestRandomSegmentLengths(
        uint32_t testsPerSegmentLength,
        uint32_t minLogTwo,
        uint32_t maxLogTwo,
        uint32_t totalSegLength,
        bool verbose)
    {
        if (k_maxTotalLength < (1 << 24) || k_maxTotalSegCount < (1 << 24))
        {
            printf("Error, allocate more memory :) \n");
            return;
        }

        printf("Beginning Split Sort Test All Random Segment Lengths \n");
        uint32_t testsPassed = 0;
        for (uint32_t maxSegLengthLog = minLogTwo; maxSegLengthLog <= maxLogTwo; ++maxSegLengthLog)
        {
            for (uint32_t i = 0; i < testsPerSegmentLength; ++i)
            {
                //Init
                uint32_t segInitInfo[2];
                if (DispatchInitSegmentsRandomLengthRandomValue(
                    segInitInfo,
                    1 << maxSegLengthLog,
                    totalSegLength,
                    BITS_TO_SORT,
                    i + 10))
                {
                    return;
                }
                cudaDeviceSynchronize();
                
                SplitSortPairs<BITS_TO_SORT>(
                    m_segments,
                    m_sort,
                    m_payloads,
                    segInitInfo[1],
                    segInitInfo[0],
                    m_tempMem);

                bool passed = ValidateSegSortRandomLength(segInitInfo[1], segInitInfo[0], true); //enable for super verbose
                if (passed)
                    testsPassed++;

                if (verbose)
                {
                    printf("Test %u: SegCount: %u TotalSegLength: %u \n", i, segInitInfo[1], segInitInfo[0]);
                    if (passed)
                        printf("Test passed.\n");
                    else
                        printf("Test failed at max seg length: %u \n", 1 << maxSegLengthLog);
                }
                else
                {
                    if ((i & 15) == 0)
                        printf(". ");
                }
                cudaDeviceSynchronize();
            }
        }

        const uint32_t testsExpected = (maxLogTwo - minLogTwo + 1) * testsPerSegmentLength;
        if (testsPassed == testsExpected)
            printf("\nSPLIT SORT ALL RANDOM SEG LENGTHS TESTS PASSED \n");
        else
            printf("\nSPLIT SORT RANDOM LENGTH TESTS FAILED %u / %u. \n", testsPassed, testsExpected);
    }

    template<uint32_t BITS_TO_SORT>
    void SanityTestRandomSegmentLengths(
        uint32_t testsPerSegmentLength,
        uint32_t minLogTwo,
        uint32_t maxLogTwo,
        uint32_t totalSegLength,
        bool verbose)
    {
        if (k_maxTotalLength < (1 << 27))
        {
            printf("Error, allocate more memory :) \n");
            return;
        }

        uint32_t* sortCopy;
        K* payloadCopy;
        cudaMalloc(&sortCopy, k_maxTotalLength * sizeof(uint32_t));
        cudaMalloc(&payloadCopy, k_maxTotalLength * sizeof(K));

        printf("Beginning Split Sort Test All Random Segment Lengths \n");
        uint32_t testsPassed = 0;
        for (uint32_t maxSegLengthLog = minLogTwo; maxSegLengthLog <= maxLogTwo; ++maxSegLengthLog)
        {
            for (uint32_t i = 0; i < testsPerSegmentLength; ++i)
            {
                //Init
                uint32_t segInitInfo[2];
                if (DispatchInitSegmentsRandomLengthRandomValue(
                    segInitInfo,
                    1 << maxSegLengthLog,
                    totalSegLength,
                    BITS_TO_SORT,
                    i + 10))
                {
                    return;
                }
                cudaDeviceSynchronize();

                cudaMemcpy(sortCopy, m_sort, segInitInfo[0] * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
                cudaMemcpy(payloadCopy, m_payloads, segInitInfo[0] * sizeof(K), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();

                //DISPATCH CUB
                void* d_temp_storage = NULL;
                size_t temp_storage_bytes = 0;
                cub::DeviceSegmentedSort::SortPairs(
                    d_temp_storage, temp_storage_bytes,
                    sortCopy, sortCopy, payloadCopy, payloadCopy,
                    segInitInfo[0], segInitInfo[1], m_segments, m_segments + 1);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceSegmentedSort::SortPairs(
                    d_temp_storage, temp_storage_bytes,
                    sortCopy, sortCopy, payloadCopy, payloadCopy,
                    segInitInfo[0], segInitInfo[1], m_segments, m_segments + 1);
                cudaFree(d_temp_storage);

                //DISPATCH SPLITSORT
                SplitSortPairs<BITS_TO_SORT>(
                    m_segments,
                    m_sort,
                    m_payloads,
                    segInitInfo[1],
                    segInitInfo[0],
                    m_tempMem);

                bool passed = ValidateSanity(sortCopy, payloadCopy, segInitInfo[1]);

                if (passed)
                    testsPassed++;

                if (verbose)
                {
                    printf("Test %u: SegCount: %u TotalSegLength: %u \n", i, segInitInfo[1], segInitInfo[0]);
                    if (passed)
                        printf("Test passed.\n");
                    else
                        printf("Test failed at max seg length: %u \n", 1 << maxSegLengthLog);
                }
                else
                {
                    if ((i & 15) == 0)
                        printf(". ");
                }
                cudaDeviceSynchronize();
            }
        }

        const uint32_t testsExpected = (maxLogTwo - minLogTwo + 1) * testsPerSegmentLength;
        if (testsPassed == testsExpected)
            printf("\nSPLIT SORT SANITY TESTS PASSED \n");
        else
            printf("\nSPLIT SORT SANITY ESTS FAILED %u / %u. \n", testsPassed, testsExpected);

        cudaFree(sortCopy);
        cudaFree(payloadCopy);
    }

    template<uint32_t BITS_TO_SORT>
    void TestFixedSegmentLength(
        uint32_t testsToRun,
        uint32_t segmentLength,
        uint32_t segmentCount,
        bool verbose)
    {
        if (k_maxTotalLength < segmentLength * segmentCount || k_maxTotalSegCount < segmentCount)
        {
            printf("Error allocate more memory\n");
            return;
        }

        printf("Beginning Split Sort: BitsToSort: %u SegmentCount: %u SegmentLength: %u TotalSegmentLength: %u \n",
            BITS_TO_SORT, segmentCount, segmentLength, segmentCount * segmentLength);
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i < testsToRun; ++i)
        {
            InitSegLengthsFixed<<<256, 256>>>(
                m_segments,
                segmentCount,
                segmentLength);
            InitFixedSegLengthRandomValue<<<4096, 64>>>(
                m_sort,
                m_payloads,
                segmentLength,
                segmentCount,
                BITS_TO_SORT,
                i + 10);
            cudaDeviceSynchronize();

            SplitSortPairs<BITS_TO_SORT>(
                m_segments,
                m_sort,
                m_payloads,
                segmentCount,
                segmentLength * segmentCount,
                m_tempMem);

            bool passed = ValidateSegSortRandomLength(segmentCount, segmentCount * segmentLength, true); //enable for super verbose
            if (passed)
                testsPassed++;

            if (verbose)
            {
                if (passed)
                    printf("Test passed.\n");
                else
                    printf("Test failed.\n");
            }
            else
            {
                if ((i & 15) == 0)
                    printf(". ");
            }
            cudaDeviceSynchronize();
        }

        if (testsPassed == testsToRun)
            printf("\nSPLIT SORT ALL FIXED SEG LENGTHS TESTS PASSED \n");
        else
            printf("\nSPLIT SORT FIXED LENGTH TESTS FAILED %u / %u. \n", testsPassed, testsToRun);
    }

    void MegaTest()
    {
        //Due to extensive use of templating, this significantly
        //increases compile times. Uncomment to enable
        /*
        TestRandomSegmentLengths<4>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<8>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<12>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<16>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<20>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<24>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<28>(100, 2, 18, 1 << 22, false);
        TestRandomSegmentLengths<32>(100, 2, 18, 1 << 22, false);

        //Test the in place OneSweep
        //1 radix pass
        TestFixedSegmentLength<4>(50, 1 << 21, 15, false);

        //2 radix pass
        TestFixedSegmentLength<8>(50, 1 << 21, 15, false);
        TestFixedSegmentLength<12>(50, 1 << 21, 15, false);

        //3 radix passes
        TestFixedSegmentLength<16>(50, 1 << 21, 15, false);
        TestFixedSegmentLength<20>(50, 1 << 21, 15, false);

        //4 radix passes
        TestFixedSegmentLength<24>(50, 1 << 21, 15, false);
        TestFixedSegmentLength<28>(50, 1 << 21, 15, false);

        //5 radix passes
        TestFixedSegmentLength<32>(50, 1 << 17, 15, false);
        TestFixedSegmentLength<32>(50, 1 << 17, 255, false);

        //6 radix passes
        TestFixedSegmentLength<32>(50, 1 << 17, 512, false);
        TestFixedSegmentLength<32>(50, 1 << 17, 1024, false);
        */
    }

    //Make a copy of the input, then use CUB to sort it.
    //Then, check if we exactly match CUB. The method of checking used above
    //will not catch some errors (all 0's for example, or a no write), so 
    //we implement this test as well
    void SanityTest()
    {
        SanityTestRandomSegmentLengths<4>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<8>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<12>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<16>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<20>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<24>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<28>(100, 2, 18, 1 << 22, false);
        SanityTestRandomSegmentLengths<32>(100, 2, 18, 1 << 22, false);
    }

private:
    //SEGINFO
    // 0 totalSegLength
    // 1 totalSegCount
    // 2: Global break flag, used for atomicCAS
    bool DispatchInitSegmentsRandomLengthRandomValue(
        uint32_t* segInitInfo,
        const uint32_t maxSegLength,
        const uint32_t maxTotalSegLength,
        const uint32_t bitsToSort,
        const uint32_t seed)
    {
        cudaMemset(m_segInitInfo, 0, 3 * sizeof(uint32_t));
        cudaDeviceSynchronize();

        //Initializing seg lengths on the CPU is slow:
        //Initialize on GPU using atomicCAS
        InitSegLengthsRandom<<<4096, 64>>>(m_segments, m_segInitInfo, seed, maxTotalSegLength, maxSegLength);
        cudaDeviceSynchronize();

        //We dont need to copy the break flag over
        cudaMemcpy(segInitInfo, m_segInitInfo, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (segInitInfo[1] + 1 > k_maxTotalSegCount || segInitInfo[0] > k_maxTotalLength)
        {
            printf("Error allocate more memory\n");
            return true;
        }

        void* d_temp_storage = NULL;
        size_t  temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            m_segments, m_segments, segInitInfo[1] + 1);    //Get the inclusive scan for CUB
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            m_segments, m_segments, segInitInfo[1] + 1);    //Get the inclusive scan for CUB
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        InitRandomSegLengthRandomValue<<<4096, 64>>>(
            m_sort,
            m_payloads,
            m_segments,
            segInitInfo[1],
            segInitInfo[0],
            bitsToSort,
            seed);

        return false;
    }
    
    bool ValidateBinning(
        uint32_t totalSegCount,
        uint32_t totalSegLength,
        uint32_t totalBinCount,
        bool verbose)
    {
        uint32_t parts = SplitSortInternal::GetNextFitPartitions(totalSegCount);
        uint32_t errCount[1];
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        ValidateBinningRandomSegLengths<<<4096, 64>>>(
            m_segments,
            SplitSortInternal::GetBinOffsetsPointer(m_tempMem, parts, totalSegCount),
            m_segInfoValidate,
            SplitSortInternal::GetPackedSegCountsPointer(m_tempMem, parts),
            m_errCount,
            totalSegCount,
            totalSegLength,
            totalBinCount,       //Because of bin packing, this is NOT the same as the totalSegCount
            verbose);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return !errCount[0];
    }

    bool ValidateSegSortRandomLength(
        uint32_t totalSegCount,
        uint32_t totalSegLength,
        bool verbose)
    {
        uint32_t errCount[1];
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        ValidateRandomLengthSegments<<<4096, 64>>>(
            m_sort,
            m_payloads,
            m_segments,
            m_errCount,
            totalSegLength,
            totalSegCount,
            verbose);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return !errCount[0];
    }

    bool ValidateSanity(
        uint32_t* sortCopy,
        K* payloadCopy,
        uint32_t totalSegLength)
    {
        uint32_t errCount[1];
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        ValidateSegSortSanity<<<256,256>>>(
            m_sort,
            sortCopy,
            m_payloads,
            payloadCopy,
            errCount,
            totalSegLength);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return !errCount[0];
    }
};

#undef SEG_INFO_SIZE

/*
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
            InitSegLengthsFixed << <256, 256 >> > (m_segments, totalSegCount, segLength);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_sort, segLength, totalSegCount);
            //InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_payloads, segLength, totalSegCount);
            InitFixedSegLengthRandomValue << <1024, 64 >> > (m_sort, m_payloads, segLength, totalSegCount, i + 10);
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
                InitSegLengthsFixed << <256, 256 >> > (m_segments, segCount, segLength);
                InitFixedSegLengthRandomValue << <1024, 64 >> > (m_sort, m_payloads, segLength, segCount, i + 10);
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
        if (size > k_maxSize || size > k_maxSegments)
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
            InitSegLengthsRandom << <4096, 64 >> > (m_segments, m_totalLength, i + 10, size, maxSegLength);
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
            InitRandomSegLengthRandomValue << <4096, 64 >> > (m_sort, m_payloads, m_segments, segInfo[1], segInfo[0], i + 10);
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
*/