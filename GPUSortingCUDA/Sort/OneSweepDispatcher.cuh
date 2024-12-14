/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/11/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../UtilityKernels.cuh"
#include "OneSweep.cuh"

#define RADIX 256
#define RADIX_LOG 8
#define SORT_PASSES 4

#define G_HIST_WARPS 4          //You can tune this
#define G_HIST_SHARED_COUNT 2   //This too
#define G_HIST_PART_SIZE 65536  //This must be a multiple of four
#define G_HIST_DIM (G_HIST_WARPS * LANE_COUNT)

#define BIN_WARPS 16            //You can tune this too
#define BIN_KEYS_PER_THREAD 15  //and this, but assert(BIN_WARPS * RADIX >= BIN_PART_SIZE)
#define BIN_DIM (BIN_WARPS * LANE_COUNT)
#define BIN_PART_SIZE (BIN_KEYS_PER_THREAD * BIN_DIM)

class OneSweepDispatcher {
    const bool k_keysOnly;
    const uint32_t k_maxSize;

    uint32_t* m_sort;
    uint32_t* m_sortPayload;
    uint32_t* m_alt;
    uint32_t* m_altPayload;
    uint32_t* m_index;
    uint32_t* m_globalHistogram;
    uint32_t* m_passHistogram;
    uint32_t* m_errCount;

   public:
    OneSweepDispatcher(bool keysOnly, uint32_t maxSize) : k_keysOnly(keysOnly), k_maxSize(maxSize) {
        assert(BIN_WARPS * RADIX >= BIN_PART_SIZE);
        const uint32_t maxBinningThreadblocks = divRoundUp(k_maxSize, BIN_PART_SIZE);
        cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_alt, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_index, SORT_PASSES * sizeof(uint32_t));
        cudaMalloc(&m_globalHistogram, SORT_PASSES * RADIX * sizeof(uint32_t));
        cudaMalloc(&m_passHistogram,
                   RADIX * SORT_PASSES * maxBinningThreadblocks * sizeof(uint32_t));
        cudaMalloc(&m_errCount, 1 * sizeof(uint32_t));
        if (!k_keysOnly) {
            cudaMalloc(&m_sortPayload, k_maxSize * sizeof(uint32_t));
            cudaMalloc(&m_altPayload, k_maxSize * sizeof(uint32_t));
        }
    }

    ~OneSweepDispatcher() {
        cudaFree(m_sort);
        cudaFree(m_alt);
        cudaFree(m_index);
        cudaFree(m_globalHistogram);
        cudaFree(m_passHistogram);
        cudaFree(m_errCount);

        if (!k_keysOnly) {
            cudaFree(m_sortPayload);
            cudaFree(m_altPayload);
        }
    }

    //Tests input sizes not perfect multiples of the partition tile size,
    //then tests several large inputs.
    void TestAllKeysOnly() {
        if (k_maxSize < (1 << 28)) {
            printf("This test requires a minimum initialized size of %u. ", 1 << 28);
            printf("Reinitialize the object to at least %u.\n", 1 << 28);
            return;
        }

        printf("Beginning GPUSorting OneSweep keys validation test: \n");
        uint32_t testsPassed = 0;
        for (uint32_t i = BIN_PART_SIZE; i < BIN_PART_SIZE * 2 + 1; ++i) {
            InitRandom<<<256, 256>>>(m_sort, ENTROPY_PRESET_1, i, i);
            DispatchKernelsKeysOnly(i);
            if (DispatchValidateKeys(i))
                testsPassed++;
            else
                printf("\n Test failed at size %u \n", i);

            if (!(i & 255))
                printf(".");
        }
        printf("\n");

        for (uint32_t i = 26; i <= 28; ++i) {
            InitRandom<<<256, 256>>>(m_sort, ENTROPY_PRESET_1, i, 1 << i);
            DispatchKernelsKeysOnly(1 << i);
            if (DispatchValidateKeys(1 << i))
                testsPassed++;
            else
                printf("\n Test failed at size %u \n", 1 << i);
        }

        if (testsPassed == BIN_PART_SIZE + 3 + 1)
            printf("%u/%u All tests passed.\n\n", testsPassed, testsPassed);
        else
            printf("%u/%u Test failed.\n\n", testsPassed, BIN_PART_SIZE + 3 + 1);
    }

    void TestAllPairs() {
        if (k_maxSize < (1 << 28)) {
            printf("This test requires a minimum initialized size of %u. ", 1 << 28);
            printf("Reinitialize the object to at least %u.\n", 1 << 28);
            return;
        }

        if (k_keysOnly) {
            printf("Error, object was intialized for keys only");
            return;
        }

        printf("Beginning GPUSorting OneSweep pairs validation test: \n");
        uint32_t testsPassed = 0;
        for (uint32_t i = BIN_PART_SIZE; i < BIN_PART_SIZE * 2 + 1; ++i) {
            InitRandom<<<256, 256>>>(m_sort, m_sortPayload, ENTROPY_PRESET_1, i, i);
            DispatchKernelsPairs(i);
            if (DispatchValidatePairs(i))
                testsPassed++;
            else
                printf("\n Test failed at size %u \n", i);

            if (!(i & 255))
                printf(".");
        }
        printf("\n");

        for (uint32_t i = 26; i <= 28; ++i) {
            InitRandom<<<256, 256>>>(m_sort, m_sortPayload, ENTROPY_PRESET_1, i, 1 << i);
            DispatchKernelsPairs(1 << i);
            if (DispatchValidatePairs(1 << i))
                testsPassed++;
            else
                printf("\n Test failed at size %u \n", 1 << i);
        }

        if (testsPassed == BIN_PART_SIZE + 3 + 1)
            printf("%u/%u All tests passed.\n\n", testsPassed, testsPassed);
        else
            printf("%u/%u Test failed.\n\n", testsPassed, BIN_PART_SIZE + 3 + 1);
    }

    void BatchTimingKeysOnly(uint32_t size, uint32_t batchCount, uint32_t seed,
                             ENTROPY_PRESET entropyPreset) {
        if (size > k_maxSize) {
            printf("Error, requested test size exceeds max initialized size. \n");
            return;
        }

        const float entLookup[5] = {1.0f, .811f, .544f, .337f, .201f};
        printf("Beginning GPUSorting OneSweep keys batch timing test at:\n");
        printf("Size: %u\n", size);
        printf("Entropy: %f bits\n", entLookup[entropyPreset]);
        printf("Test size: %u\n", batchCount);

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitRandom<<<256, 256>>>(m_sort, entropyPreset, i + seed, size);
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
        printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
               size / totalTime * batchCount);
    }

    void BatchTimingPairs(uint32_t size, uint32_t batchCount, uint32_t seed,
                          ENTROPY_PRESET entropyPreset) {
        if (k_keysOnly) {
            printf("Error, object was intialized for keys only");
            return;
        }

        if (size > k_maxSize) {
            printf("Error, requested test size exceeds max initialized size. \n");
            return;
        }

        const float entLookup[5] = {1.0f, .811f, .544f, .337f, .201f};
        printf("Beginning GPUSorting OneSweep pairs batch timing test at:\n");
        printf("Size: %u\n", size);
        printf("Entropy: %f bits\n", entLookup[entropyPreset]);
        printf("Test size: %u\n", batchCount);

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitRandom<<<256, 256>>>(m_sort, entropyPreset, i + seed, size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            DispatchKernelsPairs(size);
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
        printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
               size / totalTime * batchCount);
    }

   private:
    static inline uint32_t divRoundUp(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

    void ClearMemory(uint32_t binningThreadBlocks) {
        cudaMemset(m_index, 0, SORT_PASSES * sizeof(uint32_t));
        cudaMemset(m_globalHistogram, 0, RADIX * SORT_PASSES * sizeof(uint32_t));
        cudaMemset(m_passHistogram, 0,
                   RADIX * SORT_PASSES * binningThreadBlocks * sizeof(uint32_t));
    }

    void DispatchKernelsKeysOnly(uint32_t size) {
        const uint32_t globalHistThreadBlocks = divRoundUp(size, G_HIST_PART_SIZE);
        const uint32_t binningThreadBlocks = divRoundUp(size, BIN_PART_SIZE);
        ClearMemory(binningThreadBlocks);
        OneSweep::GlobalHistogram<uint32_t, RADIX, G_HIST_DIM, G_HIST_SHARED_COUNT,
                                  G_HIST_PART_SIZE>
            <<<globalHistThreadBlocks, G_HIST_DIM>>>(m_sort, m_globalHistogram, size);
        OneSweep::Scan<RADIX>
            <<<SORT_PASSES, RADIX>>>(m_globalHistogram, m_passHistogram, binningThreadBlocks);
        for (uint32_t k = 0; k < SORT_PASSES; ++k) {
            if (k & 1) {
                OneSweep::DigitBinningPassKeys<RADIX, RADIX_LOG, BIN_WARPS, BIN_KEYS_PER_THREAD>
                    <<<binningThreadBlocks, BIN_DIM>>>(
                        m_alt, m_sort, m_passHistogram + k * RADIX * binningThreadBlocks, m_index,
                        size, k * RADIX_LOG);
            } else {
                OneSweep::DigitBinningPassKeys<RADIX, RADIX_LOG, BIN_WARPS, BIN_KEYS_PER_THREAD>
                    <<<binningThreadBlocks, BIN_DIM>>>(
                        m_sort, m_alt, m_passHistogram + k * RADIX * binningThreadBlocks, m_index,
                        size, k * RADIX_LOG);
            }
        }
    }

    void DispatchKernelsPairs(uint32_t size) {
        const uint32_t globalHistThreadBlocks = divRoundUp(size, G_HIST_PART_SIZE);
        const uint32_t binningThreadBlocks = divRoundUp(size, BIN_PART_SIZE);
        ClearMemory(binningThreadBlocks);
        OneSweep::GlobalHistogram<uint32_t, RADIX, G_HIST_DIM, G_HIST_SHARED_COUNT,
                                  G_HIST_PART_SIZE>
            <<<globalHistThreadBlocks, G_HIST_DIM>>>(m_sort, m_globalHistogram, size);
        OneSweep::Scan<RADIX>
            <<<SORT_PASSES, RADIX>>>(m_globalHistogram, m_passHistogram, binningThreadBlocks);
        for (uint32_t k = 0; k < SORT_PASSES; ++k) {
            if (k & 1) {
                OneSweep::DigitBinningPassPairs<RADIX, RADIX_LOG, BIN_WARPS, BIN_KEYS_PER_THREAD>
                    <<<binningThreadBlocks, BIN_DIM>>>(
                        m_alt, m_altPayload, m_sort, m_sortPayload,
                        m_passHistogram + k * RADIX * binningThreadBlocks, m_index, size,
                        k * RADIX_LOG);
            } else {
                OneSweep::DigitBinningPassPairs<RADIX, RADIX_LOG, BIN_WARPS, BIN_KEYS_PER_THREAD>
                    <<<binningThreadBlocks, BIN_DIM>>>(
                        m_sort, m_sortPayload, m_alt, m_altPayload,
                        m_passHistogram + k * RADIX * binningThreadBlocks, m_index, size,
                        k * RADIX_LOG);
            }
        }
    }

    bool DispatchValidateKeys(uint32_t size) {
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        Validate<<<512, 512>>>(m_sort, m_errCount, size);
        uint32_t errCount[1];
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return !errCount[0];
    }

    bool DispatchValidatePairs(uint32_t size) {
        cudaMemset(m_errCount, 0, sizeof(uint32_t));
        Validate<<<512, 512>>>(m_sort, m_sortPayload, m_errCount, size);
        uint32_t errCount[1];
        cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return !errCount[0];
    }
};

#undef BIN_PART_SIZE
#undef BIN_DIM
#undef BIN_KEYS_PER_THREAD
#undef BIN_WARPS

#undef G_HIST_DIM
#undef G_HIST_PART_SIZE
#undef G_HIST_SHARED_COUNT
#undef G_HIST_WARPS

#undef SORT_PASSES
#undef RADIX_LOG
#undef RADIX
