#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DeviceRadixSort.cuh"
#include "UtilityKernels.cuh"

class DeviceRadixSortDispatcher
{
	const uint32_t k_maxSize;
	const uint32_t k_radix = 256;
	const uint32_t k_radixPasses = 4;
	const uint32_t k_partitionSize = 7680;
	const uint32_t k_upsweepThreads = 128;
	const uint32_t k_scanThreads = 128;
	const uint32_t k_downsweepThreads = 512;	//2080 super seems to really like 512 
	const uint32_t k_valPartSize = 4096;

	uint32_t* m_sort;
	uint32_t* m_alt;
	uint32_t* m_globalHistogram;
	uint32_t* m_passHistogram;
	uint32_t* m_errCount;

public:
	DeviceRadixSortDispatcher(uint32_t maxSize) :
		k_maxSize(maxSize)
	{
		const uint32_t threadblocks = divRoundUp(k_maxSize, k_partitionSize);
		cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
		cudaMalloc(&m_alt, k_maxSize * sizeof(uint32_t));
		cudaMalloc(&m_globalHistogram, k_radix * k_radixPasses * sizeof(uint32_t));
		cudaMalloc(&m_passHistogram, threadblocks * k_radix * sizeof(uint32_t));;
		cudaMalloc(&m_errCount, 1 * sizeof(uint32_t));
	}

	~DeviceRadixSortDispatcher()
	{
		cudaFree(m_sort);
		cudaFree(m_alt);
		cudaFree(m_globalHistogram);
		cudaFree(m_passHistogram);
		cudaFree(m_errCount);
	}

	//Tests input sizes not perfect multiples of the partition tile size,
	//then tests several large inputs.
	void TestAll()
	{
		if (k_maxSize < (1 << 28))
		{
			printf("This test requires a minimum initialized size of %u. ", 1 << 28);
			printf("Reinitialize the object to at least %u.\n", 1 << 28);
			return;
		}

		printf("Beginning validation test all: \n");
		uint32_t testsPassed = 0;
		for (uint32_t i = k_partitionSize; i < k_partitionSize * 2 + 1; ++i)
		{
			InitRandom <<<256, 256>>> (m_sort, i, i);
			DispatchKernels(i);
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
			DispatchKernels(1 << i);
			if (DispatchValidate(1 << i))
				testsPassed++;
			else
				printf("\n Test failed at size %u \n", 1 << i);
		}

		if (testsPassed == k_partitionSize + 3 + 1)
			printf("\n%u/%u All tests passed.", testsPassed, testsPassed);
		else
			printf("\n%u/%u Test failed.", testsPassed, k_partitionSize + 3 + 1);
	}

	void BatchTiming(uint32_t size, uint32_t batchCount, uint32_t seed)
	{
		if (size > k_maxSize)
		{
			printf("Error, requested test size exceeds max initialized size. \n");
			return;
		}

		printf("Beginning batch timing test at size %u and %u iterations. \n", size, batchCount);

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		float totalTime = 0.0f;
		for (uint32_t i = 0; i <= batchCount; ++i)
		{
			InitRandom <<<256, 256>>> (m_sort, size, i + seed);
			cudaDeviceSynchronize();
			cudaEventRecord(start);
			DispatchKernels(size);
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
		printf("Estimated speed at %u 32-bit elements: %E keys/sec\n", size, size / totalTime * batchCount);
	}

private:
	static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
	{
		return (x + y - 1) / y;
	}

	void DispatchKernels(uint32_t size)
	{
		const uint32_t threadblocks = divRoundUp(size, k_partitionSize);

		cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));

		cudaDeviceSynchronize();

		Upsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, size, 0);
		Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		Downsweep <<<threadblocks, k_downsweepThreads>>> (m_sort, m_alt, m_globalHistogram, m_passHistogram, size, 0);

		Upsweep <<<threadblocks, k_upsweepThreads>>> (m_alt, m_globalHistogram, m_passHistogram, size, 8);
		Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		Downsweep <<<threadblocks, k_downsweepThreads>>> (m_alt, m_sort, m_globalHistogram, m_passHistogram, size, 8);

		Upsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, size, 16);
		Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		Downsweep <<<threadblocks, k_downsweepThreads>>> (m_sort, m_alt, m_globalHistogram, m_passHistogram, size, 16);

		Upsweep <<<threadblocks, k_upsweepThreads>>> (m_alt, m_globalHistogram, m_passHistogram, size, 24);
		Scan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
		Downsweep <<<threadblocks, k_downsweepThreads>>> (m_alt, m_sort, m_globalHistogram, m_passHistogram, size, 24);
	}

	bool DispatchValidate(uint32_t size)
	{
		const uint32_t valThreadBlocks = divRoundUp(size, k_valPartSize);
		cudaMemset(m_errCount, 0, sizeof(uint32_t));
		cudaDeviceSynchronize();
		Validate <<<valThreadBlocks, 256>>> (m_sort, m_errCount, size);

		uint32_t* errCount = new uint32_t[1];
		cudaMemcpy(errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		return !errCount[0];
	}
};