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
#include "Utils.cuh"

typedef
enum ENTROPY_PRESET
{
   ENTROPY_PRESET_1 = 0,
   ENTROPY_PRESET_2 = 1,
   ENTROPY_PRESET_3 = 2,
   ENTROPY_PRESET_4 = 3,
   ENTROPY_PRESET_5 = 4,
}   ENTROPY_PRESET;

//Hybrid LCG-Tausworthe PRNG
//From GPU GEMS 3, Chapter 37
//Authors: Lee Howes and David Thomas 
#define TAUS_STEP_1         ((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19)
#define TAUS_STEP_2         ((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25)
#define TAUS_STEP_3         ((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11)
#define LCG_STEP            (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS         (z1 ^ z2 ^ z3 ^ z4)

//Initialize the input to a sequence of descending integers.
__global__ void InitDescending(uint32_t* sort, uint32_t size)
{
    for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x)
        sort[i] = size - i;
}

//An Improved Supercomputer Sorting Benchmark
//Kurt Thearling & Stephen Smith
//Bitwise AND successive keys together to decrease entropy
//in a way that is evenly distributed across histogramming
//passes.
//Number of Keys ANDed | Entropy per bit
//        0            |  1.0 bits
//        1            | .811 bits
//        2            | .544 bits
//        3            | .337 bits
//        4            | .201 bits
__global__ void InitRandom(
    uint32_t* sort,
    uint32_t andCount,
    uint32_t seed,
    uint32_t size)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;

    for (uint32_t i = idx; i < size; i += blockDim.x * gridDim.x)
    {
        uint32_t t = 0xffffffff;
        for (uint32_t k = 0; k <= andCount; ++k)
        {
            z1 = TAUS_STEP_1;
            z2 = TAUS_STEP_2;
            z3 = TAUS_STEP_3;
            z4 = LCG_STEP;
            t &= HYBRID_TAUS;
        }
        sort[i] = t;
    }
}

__global__ void InitRandom(
    uint32_t* sort,
    uint32_t* sortPayload,
    uint32_t andCount,
    uint32_t seed,
    uint32_t size)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;

    for (uint32_t i = idx; i < size; i += blockDim.x * gridDim.x)
    {
        uint32_t t = 0xffffffff;
        for (uint32_t k = 0; k <= andCount; ++k)
        {
            z1 = TAUS_STEP_1;
            z2 = TAUS_STEP_2;
            z3 = TAUS_STEP_3;
            z4 = LCG_STEP;
            t &= HYBRID_TAUS;
        }
        sort[i] = t;
        sort[i] = t;
        sortPayload[i] = t;
    }
}

#define VAL_PART_SIZE 4096
__global__ void Validate(uint32_t* sort, uint32_t* errCount, uint32_t size)
{
    __shared__ uint32_t s_val[VAL_PART_SIZE + 1];

    if (blockIdx.x < gridDim.x - 1)
    {
        const uint32_t deviceOffset = blockIdx.x * VAL_PART_SIZE;
        for (uint32_t i = threadIdx.x; i < VAL_PART_SIZE + 1; i += blockDim.x)
            s_val[i] = sort[i + deviceOffset];
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < VAL_PART_SIZE; i += blockDim.x)
        {
            if (s_val[i] > s_val[i + 1])
                atomicAdd(&errCount[0], 1);
        }
    }

    if (blockIdx.x == gridDim.x - 1)
    {
        for (uint32_t i = threadIdx.x + blockIdx.x * VAL_PART_SIZE; i < size - 1; i += blockDim.x)
        {
            if (sort[i] > sort[i + 1])
                atomicAdd(&errCount[0], 1);
        }
    }
}

//Assuming values are identical to keys, payloads must also be in sorted order
__global__ void Validate(uint32_t* sort, uint32_t* sortPayload, uint32_t* errCount, uint32_t size)
{
    __shared__ uint32_t s_val[VAL_PART_SIZE + 1];

    if (blockIdx.x < gridDim.x - 1)
    {
        const uint32_t deviceOffset = blockIdx.x * VAL_PART_SIZE;

        //Keys
        for (uint32_t i = threadIdx.x; i < VAL_PART_SIZE + 1; i += blockDim.x)
            s_val[i] = sort[i + deviceOffset];
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < VAL_PART_SIZE; i += blockDim.x)
        {
            if (s_val[i] > s_val[i + 1])
                atomicAdd(&errCount[0], 1);
        }

        //Values
        for (uint32_t i = threadIdx.x; i < VAL_PART_SIZE + 1; i += blockDim.x)
            s_val[i] = sortPayload[i + deviceOffset];
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < VAL_PART_SIZE; i += blockDim.x)
        {
            if (s_val[i] > s_val[i + 1])
                atomicAdd(&errCount[0], 1);
        }
    }

    if (blockIdx.x == gridDim.x - 1)
    {
        //keys
        for (uint32_t i = threadIdx.x + blockIdx.x * VAL_PART_SIZE; i < size - 1; i += blockDim.x)
        {
            if (sort[i] > sort[i + 1])
                atomicAdd(&errCount[0], 1);
        }

        //values
        for (uint32_t i = threadIdx.x + blockIdx.x * VAL_PART_SIZE; i < size - 1; i += blockDim.x)
        {
            if (sortPayload[i] > sortPayload[i + 1])
                atomicAdd(&errCount[0], 1);
        }
    }
}

__global__ void Print(uint32_t* toPrint, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i)
    {
        printf("%u: %u\n", i, toPrint[i]);
    }
}