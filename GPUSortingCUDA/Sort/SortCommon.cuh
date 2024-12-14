/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 12/10/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Utils.cuh"

namespace SortCommon {
    template<uint32_t KEYS_PER_THREAD, uint32_t PART_SIZE>
    __device__ __forceinline__ void LoadKeys(uint32_t* keys, uint32_t* sort, const uint32_t partitionIndex, const uint32_t size) {
        const uint32_t warpOffset = WARP_INDEX * LANE_COUNT * KEYS_PER_THREAD;
        const uint32_t devOffset = partitionIndex * PART_SIZE;
        if (partitionIndex < gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = getLaneId() + warpOffset + devOffset, k = 0; k < KEYS_PER_THREAD;
                 i += LANE_COUNT, ++k) {
                keys[k] = sort[i];
            }
        }

        if (partitionIndex == gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = getLaneId() + warpOffset + devOffset, k = 0; k < KEYS_PER_THREAD;
                 i += LANE_COUNT, ++k) {
                keys[k] = i < size ? sort[i] : 0xffffffff;
            }
        }
    }

    //Multisplitting using inline assembly and atomics
    template <uint32_t RADIX_LOG, uint32_t RADIX_MASK, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void WarpLevelMultisplit(uint32_t* keys,
                                                        const uint32_t radixShift, uint16_t* offset,
                                                        volatile uint32_t* s_warpHist) {
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
            uint32_t eqMask = 0xffffffff;
            #pragma unroll
            for (uint32_t bit = 0; bit < RADIX_LOG; ++bit) {
                const uint32_t current_bit = 1 << bit + radixShift;
                asm("{\n"
                    "    .reg .pred p;\n"
                    "    .reg .b32 bal;\n"
                    "    and.b32 bal, %1, %2;\n"
                    "    setp.ne.u32 p, bal, 0;\n"
                    "    vote.ballot.sync.b32 bal, p, 0xffffffff;\n"
                    "    @!p not.b32 bal, bal;\n"
                    "    and.b32 %0, %0, bal;\n"
                    "}\n"
                    : "+r"(eqMask)
                    : "r"(keys[k]), "r"(current_bit));
            }
            offset[k] = __popc(eqMask & getLaneMaskLt());
            const uint32_t highestRankPeer = LANE_COUNT - __clz(eqMask) - 1;
            uint32_t preIncrementVal;
            if (getLaneId() == highestRankPeer) {
                preIncrementVal = atomicAdd(
                    (uint32_t*)&s_warpHist[keys[k] >> radixShift & RADIX_MASK], offset[k] + 1);
            }
            offset[k] += __shfl_sync(0xffffffff, preIncrementVal, highestRankPeer);
        }
    }

    //Multisplitting with no assembly using barriers
    //Slower, but clearer to see what is going on
    template <uint32_t RADIX_LOG, uint32_t RADIX_MASK, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void WarpLevelMultisplitSimple(uint32_t* keys,
                                                              const uint32_t radixShift,
                                                              uint16_t* offsets,
                                                              uint32_t* s_warpHist) {
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
            uint32_t eqMask = 0xffffffff;
            #pragma unroll
            for (uint32_t bit = 0; bit < RADIX_LOG; ++bit) {
                const uint32_t current_bit = 1 << bit + radixShift;
                const bool pred = (keys[k] & current_bit) != 0;
                const uint32_t bal = __ballot_sync(0xffffffff, pred);
                eqMask &= pred ? bal : ~bal;
            }
            offsets[k] = __popc(eqMask & getLaneMaskLt());
            const uint32_t digit = keys[k] >> radixShift & RADIX_MASK;
            const uint32_t preIncrementVal = s_warpHist[digit];
            __syncwarp(0xffffffff);
            if (!offsets[k]) {
                s_warpHist[digit] += __popc(eqMask);
            }
            __syncwarp(0xffffffff);
            offsets[k] += preIncrementVal;
        }
    }

    template<uint32_t RADIX, uint32_t WARPS>
    __device__ __forceinline__ uint32_t WarpHistInclusiveScanCircularShift(uint32_t* s_warpHistograms) {
        uint32_t reduction = s_warpHistograms[threadIdx.x];
        #pragma unroll
        for (uint32_t i = threadIdx.x + RADIX, k = 0; k < WARPS - 1; i += RADIX, ++k) {
            reduction += s_warpHistograms[i];
            s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }
        return reduction;
    }

    template<uint32_t RADIX>
    __device__ __forceinline__ void WarpHistReductionExclusiveScan(uint32_t* s_localHistogram) {
        if (threadIdx.x < LANE_COUNT) {
            const uint32_t index = threadIdx.x * LANE_COUNT;
            const bool pred = index < RADIX;
            const uint32_t t = ExclusiveWarpScan(pred ? s_localHistogram[index] : 0);
            if (pred) {
                s_localHistogram[index] = t;
            }
        }
        __syncthreads();

        if (threadIdx.x < RADIX && getLaneId()) {
            s_localHistogram[threadIdx.x] += s_localHistogram[threadIdx.x - getLaneId()];
        }
    }

    template<uint32_t RADIX, uint32_t RADIX_MASK, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void UpdateOffsets(uint32_t* keys, uint16_t* offsets, uint32_t* s_warpHist,
        uint32_t* s_localHistogram, const uint32_t radixShift) {
        if (threadIdx.x >= LANE_COUNT) {
            #pragma unroll
            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
                const uint32_t t2 = keys[k] >> radixShift & RADIX_MASK;
                offsets[k] += s_warpHist[t2] + s_localHistogram[t2];
            }
        } else {
            #pragma unroll
            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
                offsets[k] += s_localHistogram[keys[k] >> radixShift & RADIX_MASK];
            }
        }
    }

    template <uint32_t RADIX, uint32_t RADIX_MASK, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void ScatterShared(uint32_t* keys, uint16_t* offsets,
        uint32_t* s_warpHistograms) {
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
            s_warpHistograms[offsets[k]] = keys[k];
        }
    }

    template <uint32_t KEYS_PER_THREAD, uint32_t RADIX_MASK>
    __device__ __forceinline__ void ScatterKeysOnly(
        uint32_t* alt, uint32_t* payload, uint32_t* altPayload, uint16_t* offsets,
        uint32_t* s_warpHistograms, uint32_t* s_localHistogram, const uint32_t radixShift,
        const uint32_t partitionIndex, const uint32_t finalPartSize) {
        if (partitionIndex < gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k) {
                alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] =
                    s_warpHistograms[i];
            }
        }

        if (partitionIndex == gridDim.x - 1) {
            for (uint32_t i = threadIdx.x; i < finalPartSize; i += blockDim.x) {
                alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] =
                    s_warpHistograms[i];
            }
        }
    }

    template <uint32_t WARPS, uint32_t KEYS_PER_THREAD, uint32_t RADIX_MASK>
    __device__ __forceinline__ void ScatterPairs(
        uint32_t* alt, uint32_t* payload, uint32_t* altPayload, uint16_t* offsets,
        uint32_t* s_warpHistograms, uint32_t* s_localHistogram, const uint32_t radixShift,
        const uint32_t partitionIndex, const uint32_t finalPartSize) {

        uint8_t digits[KEYS_PER_THREAD];
        uint32_t values[KEYS_PER_THREAD];
        if (partitionIndex < gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k) {
                digits[k] = s_warpHistograms[i] >> radixShift & RADIX_MASK;
                alt[s_localHistogram[digits[k]] + i] = s_warpHistograms[i];
            }
            __syncthreads();

            {
                const uint32_t warpOffset = WARP_INDEX * LANE_COUNT * KEYS_PER_THREAD;
                const uint32_t devOffset = partitionIndex * WARPS * LANE_COUNT * KEYS_PER_THREAD;
                #pragma unroll
                for (uint32_t i = getLaneId() + warpOffset + devOffset, k = 0; k < KEYS_PER_THREAD;
                     i += LANE_COUNT, ++k) {
                    values[k] = payload[i];
                }

                #pragma unroll
                for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
                    s_warpHistograms[offsets[k]] = values[k];
                }
            }
            __syncthreads();

            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k) {
                altPayload[s_localHistogram[digits[k]] + i] = s_warpHistograms[i];
            }
        }

        if (partitionIndex == gridDim.x - 1) {
            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k) {
                if (i < finalPartSize) {
                    digits[k] = s_warpHistograms[i] >> radixShift & RADIX_MASK;
                    alt[s_localHistogram[digits[k]] + i] = s_warpHistograms[i];
                }
            }
            __syncthreads();

            {
                const uint32_t warpOffset = WARP_INDEX * LANE_COUNT * KEYS_PER_THREAD;
                const uint32_t devOffset = partitionIndex * WARPS * LANE_COUNT * KEYS_PER_THREAD;
                #pragma unroll
                for (uint32_t i = getLaneId() + warpOffset, k = 0; k < KEYS_PER_THREAD;
                     i += LANE_COUNT, ++k) {
                    if (i < finalPartSize) {
                        values[k] = payload[i + devOffset];
                    }
                }

                #pragma unroll
                for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
                    s_warpHistograms[offsets[k]] = values[k];
                }
            }
            __syncthreads();

            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k) {
                if (i < finalPartSize) {
                    altPayload[s_localHistogram[digits[k]] + i] = s_warpHistograms[i];
                }
            }
        }
    }
}  // namespace SortCommon
