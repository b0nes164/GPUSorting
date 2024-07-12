/******************************************************************************
 * GPUSorting
 * SplitSort
 * Experimental Hybrid Radix-Merge based SegmentedSort
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 7/5/2024
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

namespace SplitSortInternal
{
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

    //Warp scans
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

    template<uint32_t N>
    struct countBits
    {
        static constexpr uint32_t count = 1 + countBits<(N >> 1)>::count;
    };

    template <>
    struct countBits<0>
    {
        static constexpr uint32_t count = 0;
    };

    template<uint32_t y>
    __host__ __forceinline__ uint32_t dvrup(uint32_t x)
    {
        constexpr uint32_t yminus = y - 1;
        constexpr uint32_t yLog = countBits<yminus>::count;
        return x + yminus >> yLog;
    }

    __host__ __forceinline__ uint32_t findHighestBit(uint32_t x)
    {
        uint32_t count = 0;
        while (x >>= 1)
            count++;

        return count;
    }
}; //Semi colon stop intellisense breaking? 