/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "EmulatedDeadlocking.cuh"
#include "UtilityKernels.cuh"

class EmulatedDeadlockingDispatcher
{
    const uint32_t k_maxSize;
    const uint32_t k_radix = 256;
    const uint32_t k_radixPasses = 4;
    const uint32_t k_partitionSize = 3840;
    const uint32_t k_globalHistPartitionSize = 65536;
    const uint32_t k_globalHistThreads = 128;
    const uint32_t k_binningThreads = 256;
    const uint32_t k_valPartSize = 4096;

    uint32_t* m_sort;
    uint32_t* m_alt;
    uint32_t* m_index;
    uint32_t* m_globalHistogram;
    uint32_t* m_firstPassHistogram;
    uint32_t* m_secPassHistogram;
    uint32_t* m_thirdPassHistogram;
    uint32_t* m_fourthPassHistogram;
    uint32_t* m_errCount;

public:
    EmulatedDeadlockingDispatcher(
        uint32_t maxSize) :
        k_maxSize(maxSize)
    {
        const uint32_t maxBinningThreadblocks = divRoundUp(k_maxSize, k_partitionSize);
        cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_alt, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_index, k_radixPasses * sizeof(uint32_t));
        cudaMalloc(&m_globalHistogram, k_radixPasses * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_firstPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_secPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_thirdPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_fourthPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_errCount, 1 * sizeof(uint32_t));
    }

    ~EmulatedDeadlockingDispatcher()
    {
        cudaFree(m_sort);
        cudaFree(m_alt);
        cudaFree(m_index);
        cudaFree(m_globalHistogram);
        cudaFree(m_firstPassHistogram);
        cudaFree(m_secPassHistogram);
        cudaFree(m_thirdPassHistogram);
        cudaFree(m_fourthPassHistogram);
        cudaFree(m_errCount);
    }

    //Tests input sizes not perfect multiples of the partition tile size,
    //then tests several large inputs.
    void TestAllKeysOnly()
    {
        if (k_maxSize < (1 << 28))
        {
            printf("This test requires a minimum initialized size of %u. ", 1 << 28);
            printf("Reinitialize the object to at least %u.\n", 1 << 28);
            return;
        }

        printf("Beginning GPUSorting OneSweep keys validation test: \n");
        uint32_t testsPassed = 0;
        for (uint32_t i = k_partitionSize; i < k_partitionSize * 2 + 1; ++i)
        {
            InitRandom << <256, 256 >> > (
                m_sort,
                ENTROPY_PRESET_1,
                i,
                i);
            DispatchKernelsKeysOnly(i);
            if (DispatchValidateKeys(i))
                testsPassed++;
            else
                printf("\n Test failed at size %u \n", i);

            if (!(i & 255))
                printf(".");
        }
        printf("\n");

        for (uint32_t i = 26; i <= 28; ++i)
        {
            InitRandom << <256, 256 >> > (
                m_sort,
                ENTROPY_PRESET_1,
                i,
                1 << i);
            DispatchKernelsKeysOnly(1 << i);
            if (DispatchValidateKeys(1 << i))
                testsPassed++;
            else
                printf("\n Test failed at size %u \n", 1 << i);
        }

        if (testsPassed == k_partitionSize + 3 + 1)
            printf("%u/%u All tests passed.\n\n", testsPassed, testsPassed);
        else
            printf("%u/%u Test failed.\n\n", testsPassed, k_partitionSize + 3 + 1);
    }

    void BatchTimingKeysOnly(uint32_t size, uint32_t batchCount, uint32_t seed, ENTROPY_PRESET entropyPreset)
    {
        if (size > k_maxSize)
        {
            printf("Error, requested test size exceeds max initialized size. \n");
            return;
        }

        const float entLookup[5] = { 1.0f, .811f, .544f, .337f, .201f };
        printf("Beginning GPUSorting OneSweep keys batch timing test at:\n");
        printf("Size: %u\n", size);
        printf("Entropy: %f bits\n", entLookup[entropyPreset]);
        printf("Test size: %u\n", batchCount);

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            InitRandom << <256, 256 >> > (
                m_sort,
                entropyPreset,
                i + seed,
                size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            DispatchKernelsKeysOnly(size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        printf("\n");
        totalTime /= 1000.0f;
        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size, size / totalTime * batchCount);
    }

private:
    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    void ClearMemory(uint32_t binningThreadBlocks)
    {
        cudaMemset(m_index, 0, k_radixPasses * sizeof(uint32_t));
        cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));
        cudaMemset(m_firstPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
        cudaMemset(m_secPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
        cudaMemset(m_thirdPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
        cudaMemset(m_fourthPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
    }

    void DispatchKernelsKeysOnly(uint32_t size)
    {
        const uint32_t globalHistThreadBlocks = divRoundUp(size, k_globalHistPartitionSize);
        const uint32_t binningThreadBlocks = divRoundUp(size, k_partitionSize);

        ClearMemory(binningThreadBlocks);

        cudaDeviceSynchronize();

        EmulatedDeadlocking::GlobalHistogram << <globalHistThreadBlocks, k_globalHistThreads >> > (m_sort, m_globalHistogram, size);

        EmulatedDeadlocking::Scan << <k_radixPasses, k_radix >> > (m_globalHistogram, m_firstPassHistogram, m_secPassHistogram,
            m_thirdPassHistogram, m_fourthPassHistogram);

        EmulatedDeadlocking::EmulatedDeadlockingSpinning << <binningThreadBlocks, k_binningThreads >> > (m_sort, m_alt, m_firstPassHistogram,
            m_index, size, 0);

        EmulatedDeadlocking::EmulatedDeadlockingSpinning << <binningThreadBlocks, k_binningThreads >> > (m_alt, m_sort, m_secPassHistogram,
            m_index, size, 8);

        EmulatedDeadlocking::EmulatedDeadlockingSpinning << <binningThreadBlocks, k_binningThreads >> > (m_sort, m_alt, m_thirdPassHistogram,
            m_index, size, 16);

        EmulatedDeadlocking::EmulatedDeadlockingSpinning << <binningThreadBlocks, k_binningThreads >> > (m_alt, m_sort, m_fourthPassHistogram,
            m_index, size, 24);
    }

    bool DispatchValidateKeys(uint32_t size)
    {
        const uint32_t valThreadBlocks = divRoundUp(size, k_valPartSize);
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        Validate << <valThreadBlocks, 256 >> > (m_sort, m_errCount, size);

        uint32_t errCount[1];
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        return !errCount[0];
    }
};