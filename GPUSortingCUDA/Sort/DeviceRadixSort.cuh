/******************************************************************************
 * GPUSorting
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 12/13/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Utils.cuh"
#include "SortCommon.cuh"

namespace DeviceRadixSort {
    template <class T, uint32_t RADIX, uint32_t BLOCK_DIM, uint32_t SHARED_HIST_COUNT,
              uint32_t PART_SIZE>
    __global__ void Upsweep(T* sort, uint32_t* globalHist, uint32_t* passHist, const uint32_t size,
                            const uint32_t digitPlace) {
        constexpr uint32_t DIGITS_PER_KEY = sizeof(T);
        constexpr uint32_t VEC_SIZE = PART_SIZE / 4;
        constexpr uint32_t DIGITS_PER_UINT4 = sizeof(uint32_t) * 4;
        constexpr uint32_t THREADS_PER_SHARED_HIST = BLOCK_DIM / SHARED_HIST_COUNT;
        constexpr uint32_t SHARED_MEM_SIZE = RADIX * SHARED_HIST_COUNT;

        __shared__ uint32_t s_mem[SHARED_MEM_SIZE];
        uint32_t* s_warpHist = &s_mem[threadIdx.x / THREADS_PER_SHARED_HIST * RADIX];

        //clear shared memory
        for (uint32_t i = threadIdx.x; i < SHARED_MEM_SIZE; i += blockDim.x) {
            s_mem[i] = 0;
        }
        __syncthreads();

        //Histogram
        if (blockIdx.x < gridDim.x - 1) {
            uint4 t[1];
            uint8_t* const bits = reinterpret_cast<uint8_t*>(t);
            const uint32_t partEnd = (blockIdx.x + 1) * VEC_SIZE;
            for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_SIZE); i < partEnd; i += blockDim.x) {
                t[0] = reinterpret_cast<uint4*>(sort)[i];
                #pragma unroll
                for (uint32_t k = 0; k < DIGITS_PER_UINT4; k += DIGITS_PER_KEY) {
                    atomicAdd(&s_warpHist[bits[k + digitPlace]], 1);
                }
            }
        }

        //Histogram
        if (blockIdx.x == gridDim.x - 1) {
            T t[1];
            uint8_t* const bits = reinterpret_cast<uint8_t*>(t);
            const uint32_t partEnd = size;
            for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < partEnd;
                 i += blockDim.x) {
                t[0] = sort[i];
                atomicAdd(&s_warpHist[bits[digitPlace]], 1);
            }
        }
        __syncthreads();

        //Reduce, pass out, prefix sum on globals, atomic add
        for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
            #pragma unroll
            for (uint32_t k = 1; k < SHARED_HIST_COUNT; ++k) {
                s_mem[i] += s_mem[i + k * RADIX];
            }
            passHist[blockIdx.x + i * gridDim.x] = s_mem[i];
            s_mem[i] = InclusiveWarpScanCircularShift(s_mem[i]);
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const uint32_t index = threadIdx.x * LANE_COUNT;
            const bool pred = index < RADIX;
            const uint32_t t = ExclusiveWarpScan(pred ? s_mem[index] : 0);
            if (pred) {
                s_mem[index] = t;
            }
        }
        __syncthreads();

        //Atomically add to device memory
        for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
            atomicAdd(&globalHist[i + digitPlace * RADIX],
                      s_mem[i] + (getLaneId() ? s_mem[i - getLaneId()] : 0));
        }
    }

    template <uint32_t WARPS, uint32_t SCAN_PER_THREAD>
    __global__ void Scan(uint32_t* passHistogram, const uint32_t threadBlocks) {
        constexpr uint32_t PART_SIZE = WARPS * LANE_COUNT * SCAN_PER_THREAD;
        __shared__ uint32_t s_red[WARPS];

        uint32_t prevReduction = 0;
        const bool lanePred = getLaneId();
        const uint32_t warpOffset = WARP_INDEX * LANE_COUNT * SCAN_PER_THREAD;
        const uint32_t digitOffset = blockIdx.x * threadBlocks;
        const uint32_t alignedSize = (threadBlocks + PART_SIZE - 1) / PART_SIZE * PART_SIZE;
        for (uint32_t devOffset = 0; devOffset < alignedSize; devOffset += PART_SIZE) {
            uint32_t scan[SCAN_PER_THREAD];
            uint32_t warpRed = 0;
            #pragma unroll
            for (uint32_t i = getLaneId() + warpOffset + devOffset, k = 0; k < SCAN_PER_THREAD;
                 i += LANE_COUNT, ++k) {
                const uint32_t t = InclusiveWarpScanCircularShift(
                    i < threadBlocks ? passHistogram[i + digitOffset] : 0);
                scan[k] = (lanePred ? t : 0) + warpRed;
                warpRed += __shfl_sync(0xffffffff, t, 0);
            }

            if (!getLaneId()) {
                s_red[WARP_INDEX] = warpRed;
            }
            __syncthreads();

            if (threadIdx.x < LANE_COUNT) {
                const bool pred = threadIdx.x < WARPS;
                const uint32_t t = InclusiveWarpScan(pred ? s_red[threadIdx.x] : 0);
                if (pred) {
                    s_red[threadIdx.x] = t;
                }
            }
            __syncthreads();

            const uint32_t totalPrev =
                (threadIdx.x >= LANE_COUNT ? s_red[WARP_INDEX - 1] : 0) + prevReduction;
            #pragma unroll
            for (uint32_t i = getLaneId() + warpOffset + devOffset, k = 0; k < SCAN_PER_THREAD;
                 i += LANE_COUNT, ++k) {
                if (i < threadBlocks) {
                    passHistogram[i + digitOffset] = scan[k] + totalPrev;
                }
            }

            prevReduction += s_red[WARPS - 1];
            __syncthreads();
        }
    }

    template <uint32_t RADIX, uint32_t RADIX_LOG, uint32_t WARPS, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void Downsweep(
        uint32_t* sort, uint32_t* payload, uint32_t* alt, uint32_t* altPayload,
        uint32_t* globalHistogram, uint32_t* passHistogram, const uint32_t size,
        const uint32_t radixShift,
        void (*Scatter)(uint32_t*, uint32_t*, uint32_t*, uint16_t*, uint32_t*, uint32_t*,
                        const uint32_t, const uint32_t, const uint32_t)) {
        constexpr uint32_t RADIX_MASK = RADIX - 1;
        constexpr uint32_t PART_SIZE = WARPS * LANE_COUNT * KEYS_PER_THREAD;
        constexpr uint32_t HISTS_SIZE = WARPS * RADIX;
        __shared__ uint32_t s_warpHistograms[PART_SIZE];
        __shared__ uint32_t s_localHistogram[RADIX];
        uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX * RADIX];

        //clear shared memory
        for (uint32_t i = threadIdx.x; i < HISTS_SIZE; i += blockDim.x)
            s_warpHistograms[i] = 0;
        __syncthreads();

        //load keys
        uint32_t keys[KEYS_PER_THREAD];
        SortCommon::LoadKeys<KEYS_PER_THREAD, PART_SIZE>(keys, sort, blockIdx.x, size);

        uint16_t offsets[KEYS_PER_THREAD];
        SortCommon::WarpLevelMultisplit<RADIX_LOG, RADIX_MASK, KEYS_PER_THREAD>(
            keys, radixShift, offsets, s_warpHist);
        __syncthreads();

        //exclusive prefix sum up the warp histograms
        if (threadIdx.x < RADIX) {
            const uint32_t reduction =
                SortCommon::WarpHistInclusiveScanCircularShift<RADIX, WARPS>(s_warpHistograms);
            //Take advantage of barrier to begin exclusive prefix sum across the reductions
            s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
        }
        __syncthreads();

        SortCommon::WarpHistReductionExclusiveScan<RADIX>(s_localHistogram);
        __syncthreads();

        SortCommon::UpdateOffsets<RADIX, RADIX_MASK, KEYS_PER_THREAD>(keys, offsets, s_warpHist,
                                                                      s_localHistogram, radixShift);
        __syncthreads();

        SortCommon::ScatterShared<RADIX, RADIX_MASK, KEYS_PER_THREAD>(keys, offsets,
                                                                      s_warpHistograms);

        if (threadIdx.x < RADIX) {
            s_localHistogram[threadIdx.x] = globalHistogram[threadIdx.x + radixShift * LANE_COUNT] +
                                            passHistogram[blockIdx.x + threadIdx.x * gridDim.x] -
                                            s_localHistogram[threadIdx.x];
        }
        __syncthreads();

        //Scatter runs of keys and/or values
        (*Scatter)(alt, payload, altPayload, offsets, s_warpHistograms, s_localHistogram,
                   radixShift, blockIdx.x, size - blockIdx.x * PART_SIZE);
    }

    template <uint32_t RADIX, uint32_t RADIX_LOG, uint32_t WARPS, uint32_t KEYS_PER_THREAD>
    __global__ void DownsweepKeys(uint32_t* sort, uint32_t* alt, uint32_t* globalHistogram,
                                  uint32_t* passHistogram, const uint32_t size,
                                  const uint32_t radixShift) {
        Downsweep<RADIX, RADIX_LOG, WARPS, KEYS_PER_THREAD>(
            sort, sort, alt, alt, globalHistogram, passHistogram, size, radixShift,
            SortCommon::ScatterKeysOnly<KEYS_PER_THREAD, RADIX - 1>);
    }

    template <uint32_t RADIX, uint32_t RADIX_LOG, uint32_t WARPS, uint32_t KEYS_PER_THREAD>
    __global__ void DownsweepPairs(uint32_t* sort, uint32_t* payload, uint32_t* alt,
                                   uint32_t* altPayload, uint32_t* globalHistogram,
                                   uint32_t* passHistogram, const uint32_t size,
                                   const uint32_t radixShift) {
        Downsweep<RADIX, RADIX_LOG, WARPS, KEYS_PER_THREAD>(
            sort, payload, alt, altPayload, globalHistogram, passHistogram, size, radixShift,
            SortCommon::ScatterPairs<WARPS, KEYS_PER_THREAD, RADIX - 1>);
    }
}  // namespace DeviceRadixSort
