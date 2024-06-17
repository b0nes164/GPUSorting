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

//General macros
#define LANE_COUNT          32							//Threads in a warp
#define LANE_MASK           31							//Mask of the lane count
#define LANE_LOG            5							//log2(LANE_COUNT)
#define WARP_INDEX          (threadIdx.x >> LANE_LOG)	//Warp of a thread

//PTX functions
__device__ __forceinline__ uint32_t getLaneId() 
{
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() 
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt()
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe()
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
    return mask;
}

//Warp scans
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

__device__ __forceinline__ uint32_t ActiveInclusiveWarpScan(uint32_t val)
{
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

__device__ __forceinline__ uint32_t ActiveInclusiveWarpScanCircularShift(uint32_t val)
{
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return __shfl_sync(mask, val, getLaneId() + LANE_MASK & LANE_MASK);
}

__device__ __forceinline__ uint32_t ExclusiveWarpScan(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    const uint32_t t = __shfl_up_sync(0xffffffff, val, 1, 32);
    return getLaneId() ? t : 0;
}

__device__ __forceinline__ uint32_t ActiveExclusiveWarpScan(uint32_t val)
{
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    const uint32_t t = __shfl_up_sync(mask, val, 1, 32);
    return getLaneId() ? t : 0;
}

__device__ __forceinline__ double WarpReduceSum(double val)
{
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) // 16 = LANE_COUNT >> 1
        val += __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);

    return val;
}

__device__ __forceinline__ uint32_t WarpReduceSum(uint32_t val)
{
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) // 16 = LANE_COUNT >> 1
        val += __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);

    return val;
}