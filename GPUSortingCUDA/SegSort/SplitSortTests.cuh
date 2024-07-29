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
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_segmented_sort.cuh"
#include "SplitSort/SplitSort.cuh"
#include "../UtilityKernels.cuh"

#define SEG_INFO_SIZE 16
#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; }

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

        cudaError_t cuda_err;
        cuda_err = cudaGetLastError();
        CUDA_CHECK(cuda_err, "Initial malloc");

        SplitSortAllocateTempMemory(k_maxTotalLength, k_maxTotalSegCount, m_tempMem);

        cuda_err = cudaGetLastError();
        CUDA_CHECK(cuda_err, "SplitSort malloc");
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

        cudaError_t cuda_err;
        cuda_err = cudaGetLastError();
        CUDA_CHECK(cuda_err, "mem free");
    }

    void TestBinningRandomSegLength(
        const uint32_t testCount,
        const uint32_t maxSegLength,
        const uint32_t totalSegLength,
        const bool verbose)
    {
        if (k_maxTotalLength < (1 << 27))
        {
            printf("Error TestBinningRandomSegLength requires at least 1 << 27 allocated sort memory.\n");
            return;
        }

        if (maxSegLength > 65536)
        {
            printf("Warning, currently a segment length greater than 65536 will skip binning, exiting.\n");
            return;
        }

        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i < testCount; ++i)
        {
            //Init
            uint32_t segInitInfo[2];
            DispatchInitSegmentsRandomLengthRandomValue(segInitInfo, maxSegLength, totalSegLength, 32, i + 10);
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

            bool passed = ValidateBinning(segInitInfo[1], segInitInfo[0], segInfo[0], true); //Enable for super verbose
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

            cudaError_t cuda_err;
            cuda_err = cudaGetLastError();
            CUDA_CHECK(cuda_err, "Binning");
        }

        if (testsPassed == testCount)
            printf("\nSPLIT SORT BINNING ALL TESTS PASSED\n");
        else
            printf("\nSPLIT SORT BINNING FAILED %u / %u \n", testsPassed, testCount);
    }

    template<uint32_t BITS_TO_SORT>
    void FastTestRandomSegmentLengths(
        uint32_t testsPerSegmentLength,
        uint32_t minLogTwo,
        uint32_t maxLogTwo,
        uint32_t totalSegLength,
        bool verbose)
    {
        if (k_maxTotalSegCount < (totalSegLength >> minLogTwo)) //Too many segments
        {
            printf("Error, allocate more memory\n");
            return;
        }

        printf("Beginning Split Sort Fast Test Random Segment Lengths \n");
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

                bool passed = ValidateSegSortRandomLength(segInitInfo[1], segInitInfo[0], false); //enable for super verbose
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

                cudaError_t cuda_err;
                cuda_err = cudaGetLastError();
                CUDA_CHECK(cuda_err, "Fast Test Random Lengths");
            }
        }

        const uint32_t testsExpected = (maxLogTwo - minLogTwo + 1) * testsPerSegmentLength;
        if (testsPassed == testsExpected)
            printf("\nSPLIT SORT FAST RANDOM SEG LENGTHS TESTS PASSED \n");
        else
            printf("\nSPLIT SORT FAST RANDOM LENGTH TESTS FAILED %u / %u. \n", testsPassed, testsExpected);
    }

    template<uint32_t BITS_TO_SORT>
    void FullTestRandomSegmentLengths(
        uint32_t testsPerSegmentLength,
        uint32_t minLogTwo,
        uint32_t maxLogTwo,
        uint32_t totalSegLength,
        bool verbose)
    {
        if (k_maxTotalSegCount < (totalSegLength >> minLogTwo)) //Too many segments
        {
            printf("Error, allocate more memory\n");
            return;
        }

        uint32_t* sortCopy;
        K* payloadCopy;
        cudaMalloc(&sortCopy, k_maxTotalLength * sizeof(uint32_t));
        cudaMalloc(&payloadCopy, k_maxTotalLength * sizeof(K));

        printf("Beginning Split Sort Full Test Random Segment Lengths \n");
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

                bool passed = ValidateFull(sortCopy, payloadCopy, segInitInfo[1]);

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
                
                cudaError_t cuda_err;
                cuda_err = cudaGetLastError();
                CUDA_CHECK(cuda_err, "Full Test Random Lengths");
            }
        }

        const uint32_t testsExpected = (maxLogTwo - minLogTwo + 1) * testsPerSegmentLength;
        if (testsPassed == testsExpected)
            printf("\nSPLIT SORT FULL TESTS PASSED \n");
        else
            printf("\nSPLIT SORT FULL TESTS FAILED %u / %u. \n", testsPassed, testsExpected);

        cudaFree(sortCopy);
        cudaFree(payloadCopy);
    }

    template<uint32_t BITS_TO_SORT>
    void FastTestFixedSegmentLength(
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

        printf("Beginning Fast Fixed Length Test: BitsToSort: %u SegmentCount: %u SegmentLength: %u TotalSegmentLength: %u \n",
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

            bool passed = ValidateSegSortRandomLength(segmentCount, segmentCount * segmentLength, false); //enable for super verbose
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
            
            cudaError_t cuda_err;
            cuda_err = cudaGetLastError();
            CUDA_CHECK(cuda_err, "Fast Fixed Test Random Lengths");
        }

        if (testsPassed == testsToRun)
            printf("\nSPLIT SORT FAST FIXED LENGTH TESTS PASSED \n");
        else
            printf("\nSPLIT SORT FAST FIXED LENGTH TESTS FAILED %u / %u. \n", testsPassed, testsToRun);
    }

    template<uint32_t BITS_TO_SORT>
    void FullTestFixedSegmentLength(
        uint32_t testsToRun,
        uint32_t segmentLength,
        uint32_t segmentCount,
        bool verbose)
    {
        if (k_maxTotalLength < segmentLength * segmentCount || k_maxTotalSegCount < segmentCount + 1)
        {
            printf("Error allocate more memory\n");
            return;
        }

        uint32_t* sortCopy;
        K* payloadCopy;
        cudaMalloc(&sortCopy, k_maxTotalLength * sizeof(uint32_t));
        cudaMalloc(&payloadCopy, k_maxTotalLength * sizeof(K));
        const uint32_t totalSegLength = segmentCount * segmentLength;

        printf("Beginning Full Fixed Length Test: BitsToSort: %u SegmentCount: %u SegmentLength: %u TotalSegmentLength: %u \n",
            BITS_TO_SORT, segmentCount, segmentLength, totalSegLength);

        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i < testsToRun; ++i)
        {
            InitSegLengthsFixed<<<256, 256>>>(
                m_segments,
                segmentCount + 1,   //For CUB
                segmentLength);
            InitFixedSegLengthRandomValue<<<4096, 64>>>(
                m_sort,
                m_payloads,
                segmentLength,
                segmentCount,
                BITS_TO_SORT,
                i + 10);
            cudaDeviceSynchronize();

            cudaMemcpy(sortCopy, m_sort, totalSegLength * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(payloadCopy, m_payloads, totalSegLength * sizeof(K), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();

            //DISPATCH CUB
            void* d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            cub::DeviceSegmentedSort::SortPairs(
                d_temp_storage, temp_storage_bytes,
                sortCopy, sortCopy, payloadCopy, payloadCopy,
                totalSegLength, segmentCount, m_segments, m_segments + 1);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedSort::SortPairs(
                d_temp_storage, temp_storage_bytes,
                sortCopy, sortCopy, payloadCopy, payloadCopy,
                totalSegLength, segmentCount, m_segments, m_segments + 1);
            cudaFree(d_temp_storage);

            SplitSortPairs<BITS_TO_SORT>(
                m_segments,
                m_sort,
                m_payloads,
                segmentCount,
                totalSegLength,
                m_tempMem);

            bool passed = ValidateFull(sortCopy, payloadCopy, totalSegLength);
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
            
            cudaError_t cuda_err;
            cuda_err = cudaGetLastError();
            CUDA_CHECK(cuda_err, "Full Fixed Test Random Lengths");
        }

        if (testsPassed == testsToRun)
            printf("\nSPLIT SORT FULL FIXED LENGTH TESTS PASSED \n");
        else
            printf("\nSPLIT SORT FULL FIXED LENGTH TESTS FAILED %u / %u. \n", testsPassed, testsToRun);

        cudaFree(sortCopy);
        cudaFree(payloadCopy);
    }

    void FastMegaTest()
    {
        //Due to extensive use of templating, this significantly
        //increases compile times. Uncomment to enable
        /*
        FastTestRandomSegmentLengths<4>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<8>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<12>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<16>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<20>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<24>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<28>(100, 2, 18, 1 << 22, false);
        FastTestRandomSegmentLengths<32>(100, 2, 18, 1 << 22, false);

        //Test the in place OneSweep
        //1 radix pass
        FastTestFixedSegmentLength<4>(50, 1 << 21, 15, false);

        //2 radix pass
        FastTestFixedSegmentLength<8>(50, 1 << 21, 15, false);
        FastTestFixedSegmentLength<12>(50, 1 << 21, 15, false);

        //3 radix passes
        FastTestFixedSegmentLength<16>(50, 1 << 21, 15, false);
        FastTestFixedSegmentLength<20>(50, 1 << 21, 15, false);

        //4 radix passes
        FastTestFixedSegmentLength<24>(50, 1 << 21, 15, false);
        FastTestFixedSegmentLength<28>(50, 1 << 21, 15, false);

        //5 radix passes
        FastTestFixedSegmentLength<32>(50, 1 << 18, 15, false);
        FastTestFixedSegmentLength<32>(50, 1 << 18, 255, false);
        
        //6 radix passes
        FastTestFixedSegmentLength<32>(50, 1 << 18, 512, false);
        */
    }

    //Make a copy of the input, then use CUB to sort it.
    //Then, check if we exactly match CUB. The method of checking used above
    //will not catch some errors (all 0's for example, or a no write), so 
    //we implement this test as well
    void FullMegaTest()
    {
        //Due to extensive use of templating, this significantly
        //increases compile times. Uncomment to enable
        FullTestRandomSegmentLengths<4>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<8>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<12>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<16>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<20>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<24>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<28>(100, 2, 18, 1 << 22, false);
        FullTestRandomSegmentLengths<32>(100, 2, 18, 1 << 22, false);

        //Test the in place OneSweep
        //1 radix pass
        FullTestFixedSegmentLength<4>(50, 1 << 21, 15, false);

        //2 radix pass
        FullTestFixedSegmentLength<8>(50, 1 << 21, 15, false);
        FullTestFixedSegmentLength<12>(50, 1 << 21, 15, false);

        //3 radix passes
        FullTestFixedSegmentLength<16>(50, 1 << 21, 15, false);
        FullTestFixedSegmentLength<20>(50, 1 << 21, 15, false);

        //4 radix passes
        FullTestFixedSegmentLength<24>(50, 1 << 21, 15, false);
        FullTestFixedSegmentLength<28>(50, 1 << 21, 15, false);

        //5 radix passes
        FullTestFixedSegmentLength<32>(50, 1 << 18, 15, false);
        FullTestFixedSegmentLength<32>(50, 1 << 18, 255, false);

        //6 radix passes
        FullTestFixedSegmentLength<32>(50, 1 << 18, 512, false);
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

        if (segInitInfo[0] > k_maxTotalLength)  //Total segment length too long
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

    bool ValidateFull(
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
            m_errCount,
            totalSegLength);
        cudaDeviceSynchronize();
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return !errCount[0];
    }
};

#undef CUDA_CHECK
#undef SEG_INFO_SIZE