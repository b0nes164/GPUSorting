#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"

__global__ void Upsweep(
	uint32_t* sort,
	uint32_t* globalHist,
	uint32_t* passHist,
	uint32_t size, 
	uint32_t radixShift);

__global__ void Scan(
	uint32_t* passHist,
	uint32_t threadBlocks);

__global__ void Downsweep(
	uint32_t* sort,
	uint32_t* alt,
	uint32_t* globalHist, 
	uint32_t* passHist,
	uint32_t size,
	uint32_t radixShift);