/******************************************************************************
 * CUB Implementations
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 ******************************************************************************/
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/agent/agent_radix_sort_onesweep.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/util_type.cuh"
#include "UtilityKernels.cuh"


template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub_t
{
	static constexpr bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

	using DominantT = cub::detail::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

	struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
	{
		static constexpr int ONESWEEP_RADIX_BITS = 8;
		static constexpr bool ONESWEEP = true;
		static constexpr bool OFFSET_64BIT = sizeof(OffsetT) == 8;

		// Onesweep policy
		using OnesweepPolicy = cub::AgentRadixSortOnesweepPolicy< 512,
			15,
			DominantT,
			1,
			cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
			cub::BLOCK_SCAN_RAKING_MEMOIZE,
			cub::RADIX_SORT_STORE_DIRECT,
			ONESWEEP_RADIX_BITS>;

		// These kernels are launched once, no point in tuning at the moment
		using HistogramPolicy =
			cub::AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
		using ExclusiveSumPolicy = cub::AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;
		using ScanPolicy = cub::AgentScanPolicy<512,
			23,
			OffsetT,
			cub::BLOCK_LOAD_WARP_TRANSPOSE,
			cub::LOAD_DEFAULT,
			cub::BLOCK_STORE_WARP_TRANSPOSE,
			cub::BLOCK_SCAN_RAKING_MEMOIZE>;

		// No point in tuning
		static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;

		// No point in tuning single-tile policy
		using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<256,
			19,
			DominantT,
			cub::BLOCK_LOAD_DIRECT,
			cub::LOAD_LDG,
			cub::RADIX_RANK_MEMOIZE,
			cub::BLOCK_SCAN_WARP_SCANS,
			SINGLE_TILE_RADIX_BITS>;
	};

	using MaxPolicy = policy_t;
};

class CubDispatcher
{
	const uint32_t k_maxSize;
	uint32_t* m_sort;
	uint32_t* m_errCount;

public:
	CubDispatcher(uint32_t size) : k_maxSize(size)
	{
		cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
		cudaMalloc(&m_errCount, sizeof(uint32_t));
	}

	~CubDispatcher()
	{
		cudaFree(m_sort);
		cudaFree(m_errCount);
	}

	void BatchTimingCubDeviceRadixSort(uint32_t size, uint32_t batchCount, uint32_t seed)
	{
		if (size > k_maxSize)
		{
			printf("Error, requested test size exceeds max initialized size. \n");
			return;
		}

		printf("Beginning batch timing test at size %u and %u iterations. \n", size, batchCount);

		void* d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
			m_sort, m_sort, size);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);

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
			cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
				m_sort, m_sort, size);
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
		cudaFree(d_temp_storage);
	}

	void BatchTimingCubOneSweep(uint32_t size, uint32_t batchCount, uint32_t seed)
	{

		if (size > k_maxSize)
		{
			printf("Error, requested test size exceeds max initialized size. \n");
			return;
		}

		printf("Beginning batch timing test at size %u and %u iterations. \n", size, batchCount);

		constexpr int begin_bit = 0;
		constexpr int end_bit = sizeof(uint32_t) * 8;
		using policy_t = policy_hub_t<uint32_t, cub::NullType, uint32_t>;
		using dispatch_t = cub::DispatchRadixSort<false, uint32_t, cub::NullType, uint32_t, policy_t>;

		cub::DoubleBuffer<uint32_t> d_keys(m_sort, m_sort);
		cub::DoubleBuffer<cub::NullType> d_values;

		void* temp_storage = NULL;
		size_t temp_storage_bytes = 0;

		dispatch_t::Dispatch(
			temp_storage,
			temp_storage_bytes,
			d_keys,
			d_values,
			size,
			begin_bit,
			end_bit,
			false,
			0);

		cudaMalloc(&temp_storage, temp_storage_bytes);

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		float totalTime = 0.0f;
		for (uint32_t i = 0; i <= batchCount; ++i)
		{
			InitRandom <<<256, 256 >>> (m_sort, size, i + seed);
			cudaDeviceSynchronize();
			cudaEventRecord(start);
			cub::DoubleBuffer<uint32_t> d_keys(m_sort, m_sort);
			cub::DoubleBuffer<cub::NullType> d_values;
			dispatch_t::Dispatch(temp_storage,
				temp_storage_bytes,
				d_keys,
				d_values,
				size,
				begin_bit,
				end_bit,
				false,
				0);
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

	void Dispatch(uint32_t size)
	{
		constexpr int begin_bit = 0;
		constexpr int end_bit = sizeof(uint32_t) * 8;
		using policy_t = policy_hub_t<uint32_t, cub::NullType, uint32_t>;
		using dispatch_t = cub::DispatchRadixSort<false, uint32_t, cub::NullType, uint32_t, policy_t>;

		cub::DoubleBuffer<uint32_t> d_keys(m_sort, m_sort);
		cub::DoubleBuffer<cub::NullType> d_values;

		void* temp_storage = NULL;
		size_t temp_storage_bytes = 0;

		dispatch_t::Dispatch(
			temp_storage,
			temp_storage_bytes,
			d_keys,
			d_values,
			size,
			begin_bit,
			end_bit,
			false,
			0);

		cudaMalloc(&temp_storage, temp_storage_bytes);

		dispatch_t::Dispatch(temp_storage,
			temp_storage_bytes,
			d_keys,
			d_values,
			size,
			begin_bit,
			end_bit,
			false,
			0);

		if (DispatchValidate(size))
			printf("Test passed.");
		else
			printf("Test failed.");
	}

private:
	bool DispatchValidate(uint32_t size)
	{
		const uint32_t valThreadBlocks = (size + 4095) / 4096;
		cudaMemset(m_errCount, 0, sizeof(uint32_t));
		cudaDeviceSynchronize();
		Validate <<<valThreadBlocks, 256 >>> (m_sort, m_errCount, size);

		uint32_t* errCount = new uint32_t[1];
		cudaMemcpy(errCount, m_errCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		return !errCount[0];
	}
};