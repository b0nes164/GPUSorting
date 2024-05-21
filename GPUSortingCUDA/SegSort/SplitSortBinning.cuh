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

namespace SplitSortBinning
{
	__global__ void NextFitBinPacking(
		const uint32_t* segments,
		uint32_t* segHist,
		uint32_t* minBinSegCounts,
		uint32_t* binOffsets,
		volatile uint32_t* index,
		volatile uint32_t* reduction,
		const uint32_t totalSegCount,
		const uint32_t totalSegLength);

	__global__ void Scan(
		uint32_t* segHist);

	__global__ void Bin(
		const uint32_t* segments,
		uint32_t* segHist,
		uint32_t* binOffsets,
		const uint32_t totalSegCount,
		const uint32_t totalSegLength);
}