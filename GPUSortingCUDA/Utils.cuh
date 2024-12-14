/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/13/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// General macros
#define LANE_COUNT 32                         // Threads in a warp
#define LANE_MASK 31                          // Mask of the lane count
#define LANE_LOG 5                            // log2(LANE_COUNT)
#define WARP_INDEX (threadIdx.x >> LANE_LOG)  // Warp of a thread

// PTX functions
__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

// Warp scans
template <class T>
__device__ __forceinline__ T InclusiveWarpScan(T val) {
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)  { // 16 = LANE_COUNT >> 1
        const T t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) {
            val += t;
        }
    }
    return val;
}

template <class T>
__device__ __forceinline__ T InclusiveWarpScanCircularShift(T val) {
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)  { // 16 = LANE_COUNT >> 1
        const T t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) {
            val += t;
        }
    }
    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

template <class T>
__device__ __forceinline__ T ExclusiveWarpScan(T val) {
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)  { // 16 = LANE_COUNT >> 1
        const T t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) {
            val += t;
        }
    }
    const T t = __shfl_up_sync(0xffffffff, val, 1, 32);
    return getLaneId() ? t : 0;
}

template <class T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) { // 16 = LANE_COUNT >> 1
        val += __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);
    }  
    return val;
}
