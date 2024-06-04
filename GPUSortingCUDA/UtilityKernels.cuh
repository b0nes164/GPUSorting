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
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

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
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

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
        sortPayload[i] = t;
    }
}

//Kernels for Segmented Sort testing:
//Create descending sequences of exact length of a segment
__global__ void InitFixedSegLengthDescendingValue(
    uint32_t* sort,
    uint32_t segLength,
    uint32_t totalSegCount)
{
    const uint32_t sCount = totalSegCount;
    const uint32_t sLength = segLength;

    for (uint32_t k = blockIdx.x; k < sCount; k += gridDim.x)
    {
        const uint32_t devOffset = k * sLength;
        for (uint32_t i = threadIdx.x; i < sLength; i += blockDim.x)
            sort[i + devOffset] = sLength - i;
    }
}

__global__ void InitFixedSegLengthRandomValue(
    uint32_t* sort,
    uint32_t* payload,
    uint32_t segLength,
    uint32_t totalSegCount,
    uint32_t seed)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

    const uint32_t sCount = totalSegCount;
    const uint32_t sLength = segLength;
    for (uint32_t k = blockIdx.x; k < sCount; k += gridDim.x)
    {
        const uint32_t devOffset = k * sLength;
        for (uint32_t i = threadIdx.x; i < sLength; i += blockDim.x)
        {
            z1 = TAUS_STEP_1;
            z2 = TAUS_STEP_2;
            z3 = TAUS_STEP_3;
            z4 = LCG_STEP;
            const uint32_t t = HYBRID_TAUS;
            sort[i + devOffset] = t;
            payload[i + devOffset] = t;
        }
    }
}

__global__ void InitFixedSegLengthRandomValue(
    uint32_t* sort,
    double* payload,
    uint32_t segLength,
    uint32_t totalSegCount,
    uint32_t seed)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

    const uint32_t sCount = totalSegCount;
    const uint32_t sLength = segLength;
    for (uint32_t k = blockIdx.x; k < sCount; k += gridDim.x)
    {
        const uint32_t devOffset = k * sLength;
        for (uint32_t i = threadIdx.x; i < sLength; i += blockDim.x)
        {
            const uint32_t t = HYBRID_TAUS;
            sort[i + devOffset] = t;

            z1 = TAUS_STEP_1;
            z2 = TAUS_STEP_2;
            z3 = TAUS_STEP_3;
            z4 = LCG_STEP;

            uint64_t y = (uint64_t)HYBRID_TAUS << 32 | t;
            //uint64_t y = (uint64_t)t;
            double x;
            memcpy(&x, &y, sizeof(double));
            payload[i + devOffset] = x;
        }
    }
}

__global__ void InitRandomSegLengthRandomValue(
    uint32_t* sort,
    uint32_t* payload,
    uint32_t* segments,
    uint32_t totalSegCount,
    uint32_t totalSegLength,
    uint32_t seed)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

    const uint32_t sCount = totalSegCount;
    for (uint32_t k = blockIdx.x; k < sCount; k += gridDim.x)
    {
        const uint32_t segmentStart = segments[k];
        const uint32_t segmentEnd = k + 1 == totalSegCount ? totalSegLength : segments[k + 1];
        const uint32_t segLength = segmentEnd - segmentStart;
        for (uint32_t i = threadIdx.x; i < segLength; i += blockDim.x)
        {
            z1 = TAUS_STEP_1;
            z2 = TAUS_STEP_2;
            z3 = TAUS_STEP_3;
            z4 = LCG_STEP;
            const uint32_t t = HYBRID_TAUS;
            sort[i + segmentStart] = t;
            payload[i + segmentStart] = t;
        }
    }
}

__global__ void InitRandomSegLengthUniqueValue(
    uint32_t* sort,
    uint32_t* payload,
    uint32_t* segments,
    uint32_t totalSegCount,
    uint32_t totalSegLength,
    uint32_t seed)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

    for (uint32_t block = blockIdx.x; block < totalSegCount; block += gridDim.x)
    {
        const uint32_t segmentStart = segments[block];
        const uint32_t segmentEnd = block + 1 == totalSegCount ? totalSegLength : segments[block + 1];
        const uint32_t segLength = segmentEnd - segmentStart;
        __shared__ uint32_t s_mem[4096];
        if (segLength <= 4096)
        {
            for (uint32_t i = threadIdx.x; i < segLength; i += blockDim.x)
                s_mem[i] = segLength - i;
            __syncthreads();

            #pragma unroll
            for (uint32_t t = 0; t < 2; ++t)
            {
                #pragma unroll
                for (uint32_t i = 1; i < 3; ++i)
                {
                    uint32_t part = segLength >> i;
                    for (uint32_t j = 0; j < i; ++j)
                    {
                        for (uint32_t k = threadIdx.x; k < part; k += blockDim.x)
                        {
                            z1 = TAUS_STEP_1;
                            z2 = TAUS_STEP_2;
                            z3 = TAUS_STEP_3;
                            z4 = LCG_STEP;

                            uint32_t rand = HYBRID_TAUS;
                            uint32_t index = k + j * part * 2;
                            if (rand < 0x80000000 && index < segLength)
                            {
                                uint32_t t = s_mem[index];
                                s_mem[index] = s_mem[index + part];
                                s_mem[index + part] = t;
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            for (uint32_t i = threadIdx.x; i < segLength; i += blockDim.x)
            {
                sort[i + segmentStart] = s_mem[i];
                payload[i + segmentStart] = s_mem[i];
            }
            __syncthreads();
        }

        if (segLength > 4096)
        {
            //direct device memory here, unnecessary for now as we dont test > 4096
        }
    }
}

//Because seg lengths are fixed, we can skip prefix sum
//by multiplying the index by the seg length
__global__ void InitSegLengthsFixed(
    uint32_t* segments,
    uint32_t maxSegments,
    uint32_t segmentLength)
{
    const uint32_t segLength = segmentLength;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < maxSegments; i += blockDim.x * gridDim.x)
        segments[i] = i * segLength;
}

//Initialize segment lengths, with each segment length 
//limited to maxSegLength, and the total length of segments limited to maxTotalLength
__global__ void InitSegLengthsRandom(
    uint32_t* segments,
    volatile uint32_t* totalLength,
    uint32_t seed,
    uint32_t maxTotalLength,
    uint32_t maxSegLength)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t z1 = (idx << 2) * seed;
    uint32_t z2 = ((idx << 2) + 1) * seed;
    uint32_t z3 = ((idx << 2) + 2) * seed;
    uint32_t z4 = ((idx << 2) + 3) * seed;
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;
    do 
    {
        z1 = TAUS_STEP_1;
        z2 = TAUS_STEP_2;
        z3 = TAUS_STEP_3;
        z4 = LCG_STEP;
        uint32_t t = HYBRID_TAUS % maxSegLength + 1;
        const uint32_t reduce = WarpReduceSum(t);
        bool validAddition = false;

        if (!getLaneId())
        {
            uint32_t assumed;
            uint32_t old = 0xffffffff;
            do
            {
                __threadfence();
                assumed = totalLength[0];
                if (assumed + reduce < maxTotalLength)  //ok to add
                {
                    old = atomicCAS((uint32_t*)&totalLength[0], assumed, assumed + reduce);
                    if(assumed == old)
                        validAddition = true;
                }
                else //too much, set the global break flag
                {
                    totalLength[2] = 1;
                    break;
                }

            } while (assumed != old && totalLength[2] == 0);
        }
        __syncwarp(0xffffffff);

        if (__shfl_sync(0xffffffff, validAddition, 0))
        {
            uint32_t deviceOffset;
            if (!getLaneId())
                deviceOffset = atomicAdd((uint32_t*)&totalLength[1], LANE_COUNT);
            deviceOffset = __shfl_sync(0xffffffff, deviceOffset, 0);
            segments[getLaneId() + deviceOffset] = t;
        }
    } while (totalLength[2] == 0);
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

__global__ void ValidateFixLengthSegments(
    uint32_t* sort,
    uint32_t* payload,
    uint32_t* errCount,
    uint32_t segLength,
    uint32_t totalSegCount)
{
    const uint32_t sCount = totalSegCount;
    const uint32_t sLength = segLength;

    for (uint32_t k = blockIdx.x; k < sCount; k += gridDim.x)
    {
        const uint32_t devOffset = k * sLength;
        for (uint32_t i = threadIdx.x + 1; i < sLength; i += blockDim.x)
        {
            if(sort[i + devOffset - 1] > sort[i + devOffset])
                atomicAdd((uint32_t*)&errCount[0], 1);

            if (payload[i + devOffset - 1] > payload[i + devOffset])
                atomicAdd((uint32_t*)&errCount[0], 1);
        }
    }
}

__global__ void ValidateFixLengthSegments(
    uint32_t* sort,
    double* payload,
    uint32_t* errCount,
    uint32_t segLength,
    uint32_t totalSegCount)
{
    const uint32_t sCount = totalSegCount;
    const uint32_t sLength = segLength;

    for (uint32_t k = blockIdx.x; k < sCount; k += gridDim.x)
    {
        const uint32_t devOffset = k * sLength;
        for (uint32_t i = threadIdx.x + 1; i < sLength; i += blockDim.x)
        {
            if (sort[i + devOffset - 1] > sort[i + devOffset])
                atomicAdd((uint32_t*)&errCount[0], 1);

            double d1 = payload[i + devOffset - 1];
            double d2 = payload[i + devOffset];

            uint32_t u1;
            uint32_t u2;
            memcpy(&u1, &d1, sizeof(uint32_t));
            memcpy(&u2, &d2, sizeof(uint32_t)); //Copy the lower 32 bits, which match the keys as uints

            //If the payloads were moved correctly,
            //They must also be in sorted order
            if (u1 > u2)
                atomicAdd((uint32_t*)&errCount[0], 1);
        }
    }
}

__global__ void ValidateRandomLengthSegments(
    uint32_t* sort,
    uint32_t* payload,
    uint32_t* segments,
    uint32_t* errCount,
    uint32_t totalSegLength,
    uint32_t totalSegCount)
{
    for (uint32_t k = blockIdx.x; k < totalSegCount; k += gridDim.x)
    {
        const uint32_t segmentStart = segments[k];
        const uint32_t segmentEnd = k + 1 == totalSegCount ? totalSegLength : segments[k + 1];
        const uint32_t segLength = segmentEnd - segmentStart;

        for (uint32_t i = threadIdx.x + 1; i < segLength; i += blockDim.x)
        {
            if (sort[i + segmentStart - 1] > sort[i + segmentStart])
            {
                atomicAdd((uint32_t*)&errCount[0], 1);
                /*printf("Sort error: %u %u. Segment Index: %u. Segment length %u.\n",
                    sort[i + segmentStart - 1], sort[i + segmentStart], k, segLength);*/
            }
                

            if (payload[i + segmentStart - 1] > payload[i + segmentStart])
            {
                atomicAdd((uint32_t*)&errCount[0], 1);
                /*printf("Payload error: %u %u. Segment Index: %u. Segment length %u.\n",
                    payload[i + segmentStart - 1], payload[i + segmentStart], k, segLength);*/
            }
                
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