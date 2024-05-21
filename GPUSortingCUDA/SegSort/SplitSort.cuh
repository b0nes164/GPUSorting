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
#include "../Utils.cuh"

namespace SplitSort 
{
	template<uint32_t BITS_TO_SORT>
	__global__ void SortLe32(
		const uint32_t* segments,
		const uint32_t* binOffsets,
		const uint32_t* minBinSegCounts,
		uint32_t* sort,
		uint32_t* payloads);

	template<uint32_t BITS_TO_SORT>
	__global__ void SortGt32Le64(
		const uint32_t* segments,
		const uint32_t* binOffset,
		uint32_t* sort,
		uint32_t* payloads);
}