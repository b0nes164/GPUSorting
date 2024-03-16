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
#include "DeviceRadixSort.cuh"
#include "UtilityKernels.cuh"

class DeviceRadixSortDispatcher
{
	const bool k_keysOnly;
	const uint32_t k_maxSize;
	const uint32_t k_radix = 256;
	const uint32_t k_radixPasses = 4;
	const uint32_t k_partitionSize = 7680;
	const uint32_t k_upsweepThreads = 128;
	const uint32_t k_scanThreads = 128;
	const uint32_t k_downsweepThreads = 512;	//2080 super seems to really like 512 
	const uint32_t k_valPartSize = 4096;

	uint32_t* m_sort;
	uint32_t* m_sortPayload;
	uint32_t* m_alt;
	uint32_t* m_altPayload;
	uint32_t* m_globalHistogram;
	uint32_t* m_passHistogram;
	uint32_t* m_errCount;

public:
	DeviceRadixSortDispatcher(
		bool keysOnly,
		uint32_t maxSize) :
		k_keysOnly(keysOnly),
		k_maxSize(maxSize)
	{
		const uint32_t threadblocks = divRoundUp(k_maxSize, k_partitionSize);
		cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
		cudaMalloc(&m_alt, k_maxSize * sizeof(uint32_t));
		cudaMalloc(&m_globalHistogram, k_radix * k_radixPasses * sizeof(uint32_t));
		cudaMalloc(&m_passHistogram, threadblocks * k_radix * sizeof(uint32_t));;
		cudaMalloc(&m_errCount, 1 * sizeof(uint32_t));

		if (!k_keysOnly)
		{
			cudaMalloc(&m_sortPayload, k_maxSize * sizeof(uint32_t));
			cudaMalloc(&m_altPayload, k_maxSize * sizeof(uint32_t));
		}
	}

	~DeviceRadixSortDispatcher()
	{
		cudaFree(m_sort);
		cudaFree(m_alt);
		cudaFree(m_globalHistogram);
		cudaFree(m_passHistogram);
		cudaFree(m_errCount);

		if (!k_keysOnly)
		{
			cudaFree(m_sortPayload);
			cudaFree(m_altPayload);
		}
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

		printf("Beginning GPUSorting DeviceRadixSort keys validation test: \n");
		uint32_t testsPassed = 0;
		for (uint32_t i = k_partitionSize; i < k_partitionSize * 2 + 1; ++i)
		{
			InitRandom <<<256, 256>>> (m_sort, i, i);
			DispatchKernelsKeysOnly(i);
			if (DispatchValidate(i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", i);

			if (!(i & 255))
				printf(".");
		}
		printf("\n");

		for (uint32_t i = 26; i <= 28; ++i)
		{
			InitRandom <<<256, 256>>> (m_sort, 1 << i, 5);
			DispatchKernelsKeysOnly(1 << i);
			if (DispatchValidate(1 << i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", 1 << i);
		}

		if (testsPassed == k_partitionSize + 3 + 1)
			printf("%u/%u All tests passed.\n\n", testsPassed, testsPassed);
		else
			printf("%u/%u Test failed.\n\n", testsPassed, k_partitionSize + 3 + 1);
	}

	void TestAllPairs()
	{
		if (k_maxSize < (1 << 28))
		{
			printf("This test requires a minimum initialized size of %u. ", 1 << 28);
			printf("Reinitialize the object to at least %u.\n", 1 << 28);
			return;
		}

		if (k_keysOnly)
		{
			printf("Error, object was intialized for keys only");
			return;
		}

		printf("Beginning GPUSorting DeviceRadixSort pairs validation test: \n");
		uint32_t testsPassed = 0;
		for (uint32_t i = k_partitionSize; i < k_partitionSize * 2 + 1; ++i)
		{
			InitRandom <<<256, 256>>> (m_sort, m_sortPayload, i, i);
			DispatchKernelsPairs(i);
			if (DispatchValidatePairs(i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", i);

			if (!(i & 255))
				printf(".");
		}
		printf("\n");

		for (uint32_t i = 26; i <= 28; ++i)
		{
			InitRandom <<<256, 256>>> (m_sort, m_sortPayload, 1 << i, 5);
			DispatchKernelsPairs(1 << i);
			if (DispatchValidatePairs(1 << i))
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
		printf("Beginning GPUSorting DeviceRadixSort keys batch timing test at:\n");
		printf("Size: %u\n", size);
		printf("Entropy: %f bits\n", entLookup[entropyPreset - 1]);
		printf("Test size: %u\n", batchCount);

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		float totalTime = 0.0f;
		for (uint32_t i = 0; i <= batchCount; ++i)
		{
			InitRandom <<<256, 256 >>> (m_sort, size, i + seed);
			if (entropyPreset > ENTROPY_PRESET_1)
				InitEntropyControlled <<<256, 256 >>> (m_sort, entropyPreset, size);
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

	void BatchTimingPairs(uint32_t size, uint32_t batchCount, uint32_t seed, ENTROPY_PRESET entropyPreset)
	{
		if (size > k_maxSize)
		{
			printf("Error, requested test size exceeds max initialized size. \n");
			return;
		}

		if (k_keysOnly)
		{
			printf("Error, object was intialized for keys only");
			return;
		}

		const float entLookup[5] = { 1.0f, .811f, .544f, .337f, .201f };
		printf("Beginning GPUSorting DeviceRadixSort pairs batch timing test at:\n");
		printf("Size: %u\n", size);
		printf("Entropy: %f bits\n", entLookup[entropyPreset - 1]);
		printf("Test size: %u\n", batchCount);

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		float totalTime = 0.0f;
		for (uint32_t i = 0; i <= batchCount; ++i)
		{
			InitRandom <<<256, 256 >>> (m_sort, size, i + seed);
			if (entropyPreset > ENTROPY_PRESET_1)
				InitEntropyControlled <<<256, 256 >>> (m_sort, entropyPreset, size);
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
		printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size, size / totalTime * batchCount);
	}

private:
	static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
	{
		return (x + y - 1) / y;
	}

	void DispatchKernelsKeysOnly(uint32_t size)
	{
		const uint32_t threadblocks = divRoundUp(size, k_partitionSize);

		cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));

		cudaDeviceSynchronize();

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, size, 0);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepKeysOnly <<<threadblocks, k_downsweepThreads>>> (m_sort, m_alt, m_globalHistogram, m_passHistogram, size, 0);

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_alt, m_globalHistogram, m_passHistogram, size, 8);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepKeysOnly <<<threadblocks, k_downsweepThreads>>> (m_alt, m_sort, m_globalHistogram, m_passHistogram, size, 8);

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, size, 16);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepKeysOnly <<<threadblocks, k_downsweepThreads>>> (m_sort, m_alt, m_globalHistogram, m_passHistogram, size, 16);

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_alt, m_globalHistogram, m_passHistogram, size, 24);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepKeysOnly <<<threadblocks, k_downsweepThreads>>> (m_alt, m_sort, m_globalHistogram, m_passHistogram, size, 24);
	}

	void DispatchKernelsPairs(uint32_t size)
	{
		const uint32_t threadblocks = divRoundUp(size, k_partitionSize);

		cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));

		cudaDeviceSynchronize();

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, size, 0);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepPairs <<<threadblocks, k_downsweepThreads>>> (m_sort, m_sortPayload, m_alt, m_altPayload,
			m_globalHistogram, m_passHistogram, size, 0);

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_alt, m_globalHistogram, m_passHistogram, size, 8);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepPairs <<<threadblocks, k_downsweepThreads>>> (m_alt, m_altPayload, m_sort, m_sortPayload,
			m_globalHistogram, m_passHistogram, size, 8);

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, size, 16);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepPairs <<<threadblocks, k_downsweepThreads>>> (m_sort, m_sortPayload, m_alt, m_altPayload,
			m_globalHistogram, m_passHistogram, size, 16);

		DeviceRadixSort::Upsweep <<<threadblocks, k_upsweepThreads>>> (m_alt, m_globalHistogram, m_passHistogram, size, 24);
		DeviceRadixSort::Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		DeviceRadixSort::DownsweepPairs <<<threadblocks, k_downsweepThreads>>> (m_alt, m_altPayload, m_sort, m_sortPayload,
			m_globalHistogram, m_passHistogram, size, 24);
	}

	bool DispatchValidate(uint32_t size)
	{
		const uint32_t valThreadBlocks = divRoundUp(size, k_valPartSize);
		cudaMemset(m_errCount, 0, sizeof(uint32_t));
		cudaDeviceSynchronize();
		Validate <<<valThreadBlocks, 256>>> (m_sort, m_errCount, size);

		uint32_t errCount[1];
		cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		return !errCount[0];
	}

	bool DispatchValidatePairs(uint32_t size)
	{
		const uint32_t valThreadBlocks = divRoundUp(size, k_valPartSize);
		cudaMemset(m_errCount, 0, sizeof(uint32_t));
		cudaDeviceSynchronize();
		Validate <<<valThreadBlocks, 256 >>> (m_sort, m_sortPayload, m_errCount, size);

		uint32_t errCount[1];
		cudaMemcpy(&errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		return !errCount[0];
	}
};