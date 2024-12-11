/******************************************************************************
 * GPUSorting
 * OneSweep Implementation
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 12/10/2024
 * https://github.com/b0nes164/GPUSorting
 *
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include "../Utils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY \
    0  //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION 1  //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE 2  //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK 3       //Mask used to retrieve flag values

namespace OneSweep {
    template <class T, uint32_t RADIX, uint32_t BLOCK_DIM, uint32_t SHARED_HIST_COUNT,
              uint32_t HIST_PART_SIZE>
    __global__ void GlobalHistogram(T* sort, uint32_t* globalHistogram, uint32_t size) {
        constexpr uint32_t HIST_VEC_SIZE = HIST_PART_SIZE / 4;
        constexpr uint32_t DIGITS_PER_KEY = sizeof(T);
        constexpr uint32_t DIGITS_PER_UINT4 = sizeof(uint32_t) * 4;
        constexpr uint32_t THREADS_PER_SHARED_HIST = BLOCK_DIM / SHARED_HIST_COUNT;
        constexpr uint32_t SHARED_MEM_SIZE = RADIX * SHARED_HIST_COUNT * DIGITS_PER_KEY;
        __shared__ uint32_t s_mem[SHARED_MEM_SIZE];
        uint32_t* s_warpHist[DIGITS_PER_KEY];

        #pragma unroll
        for (uint32_t i = 0; i < DIGITS_PER_KEY; ++i) {
            s_warpHist[i] =
                &s_mem[(i + threadIdx.x / THREADS_PER_SHARED_HIST * DIGITS_PER_KEY) * RADIX];
        }

        //clear shared memory
        for (uint32_t i = threadIdx.x; i < SHARED_MEM_SIZE; i += blockDim.x) {
            s_mem[i] = 0;
        }
        __syncthreads();

        if (blockIdx.x < gridDim.x - 1) {
            uint4 t[1];
            uint8_t* const bits = reinterpret_cast<uint8_t*>(t);
            const uint32_t partEnd = (blockIdx.x + 1) * HIST_VEC_SIZE;
            for (uint32_t i = threadIdx.x + (blockIdx.x * HIST_VEC_SIZE); i < partEnd;
                 i += blockDim.x) {
                t[0] = reinterpret_cast<uint4*>(sort)[i];
                #pragma unroll
                for (uint32_t k = 0; k < DIGITS_PER_UINT4; k += DIGITS_PER_KEY) {
                    #pragma unroll
                    for (uint32_t j = 0; j < DIGITS_PER_KEY; ++j) {
                        atomicAdd(&s_warpHist[j][bits[k + j]], 1);
                    }
                }
            }
        }

        if (blockIdx.x == gridDim.x - 1) {
            T t[1];
            uint8_t* const bits = reinterpret_cast<uint8_t*>(t);
            const uint32_t partEnd = size;
            for (uint32_t i = threadIdx.x + (blockIdx.x * HIST_PART_SIZE); i < partEnd;
                 i += blockDim.x) {
                t[0] = sort[i];
                #pragma unroll
                for (uint32_t k = 0; k < DIGITS_PER_KEY; ++k) {
                    atomicAdd(&s_warpHist[k][bits[k]], 1);
                }
            }
        }
        __syncthreads();

        //reduce and add to device
        #pragma unroll
        for (uint32_t k = 0; k < DIGITS_PER_KEY; ++k) {
            for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
                uint32_t t = 0;
                #pragma unroll
                for (uint32_t j = 0; j < SHARED_HIST_COUNT; ++j) {
                    t += s_mem[i + (k + j * DIGITS_PER_KEY) * RADIX];
                }
                atomicAdd(&globalHistogram[i + k * RADIX], t);
            }
        }
    }

    template <uint32_t RADIX>
    __global__ void Scan(uint32_t* globalHistogram, uint32_t* passHistogram,
                         const uint32_t threadBlocks) {
        constexpr uint32_t SPINE_SIZE = RADIX / LANE_COUNT;
        __shared__ uint32_t s_scan[SPINE_SIZE];

        const uint32_t scan = globalHistogram[threadIdx.x + blockIdx.x * RADIX];
        const uint32_t reduce = WarpReduceSum(scan);
        if (!getLaneId()) {
            s_scan[WARP_INDEX] = reduce;
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const bool pred = threadIdx.x < SPINE_SIZE;
            const uint32_t t = ExclusiveWarpScan(pred ? s_scan[threadIdx.x] : 0);
            if (pred) {
                s_scan[threadIdx.x] = t;
            }
        }
        __syncthreads();

        const uint32_t passIndex = threadIdx.x + blockIdx.x * threadBlocks * RADIX;
        passHistogram[passIndex] =
            ExclusiveWarpScan(scan) + s_scan[WARP_INDEX] << 2 | FLAG_INCLUSIVE;
    }

    //Multisplitting using inline assembly and atomics
    template <uint32_t RADIX_LOG, uint32_t RADIX_MASK>
    __device__ __forceinline__ void WarpLevelMultisplit(const uint32_t key,
                                                        const uint32_t radixShift, uint16_t& offset,
                                                        volatile uint32_t* s_warpHist) {
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
                : "r"(key), "r"(current_bit));
        }
        offset = __popc(eqMask & getLaneMaskLt());
        const uint32_t highestRankPeer = LANE_COUNT - __clz(eqMask) - 1;
        uint32_t preIncrementVal;
        if (getLaneId() == highestRankPeer)
            preIncrementVal =
                atomicAdd((uint32_t*)&s_warpHist[key >> radixShift & RADIX_MASK], offset + 1);
        offset += __shfl_sync(0xffffffff, preIncrementVal, highestRankPeer);
    }

    //Multisplitting with no assembly using barriers
    //Slower, but clearer to see what is going on
    template <uint32_t RADIX_LOG, uint32_t RADIX_MASK>
    __device__ __forceinline__ void WarpLevelMultisplitSimple(const uint32_t key,
                                                              const uint32_t radixShift,
                                                              uint16_t& offset,
                                                              uint32_t* s_warpHist) {
        uint32_t eqMask = 0xffffffff;
        #pragma unroll
        for (uint32_t bit = 0; bit < RADIX_LOG; ++bit) {
            const uint32_t current_bit = 1 << bit + radixShift;
            const bool pred = (key & current_bit) != 0;
            const uint32_t bal = __ballot_sync(0xffffffff, pred);
            eqMask &= pred ? bal : ~bal;
        }
        offset = __popc(eqMask & getLaneMaskLt());
        const uint32_t digit = key >> radixShift & RADIX_MASK;
        const uint32_t preIncrementVal = s_warpHist[digit];
        __syncwarp(0xffffffff);
        if (!offset) {
            s_warpHist[digit] += __popc(eqMask);
        }
        __syncwarp(0xffffffff);
        offset += preIncrementVal;
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

    template <uint32_t RADIX, uint32_t RADIX_LOG, uint32_t WARPS, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void DigitBinningPass(
        uint32_t* sort, uint32_t* payload, uint32_t* alt, uint32_t* altPayload,
        volatile uint32_t* passHistogram, volatile uint32_t* index, const uint32_t size,
        const uint32_t radixShift,
        void (*Scatter)(uint32_t*, uint32_t*, uint32_t*, uint16_t*, uint32_t*, uint32_t*, uint32_t,
                        uint32_t, uint32_t)) {
        constexpr uint32_t RADIX_MASK = RADIX - 1;
        constexpr uint32_t PART_SIZE = WARPS * LANE_COUNT * KEYS_PER_THREAD;
        constexpr uint32_t HISTS_SIZE = WARPS * RADIX;
        __shared__ uint32_t s_warpHistograms[PART_SIZE];
        __shared__ uint32_t s_localHistogram[RADIX];
        volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX * RADIX];

        //clear shared memory
        for (uint32_t i = threadIdx.x; i < HISTS_SIZE; i += blockDim.x)
            s_warpHistograms[i] = 0;

        //atomically assign partition tiles
        if (threadIdx.x == 0)
            s_localHistogram[0] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
        __syncthreads();
        const uint32_t partitionIndex = s_localHistogram[0];

        //load keys
        uint32_t keys[KEYS_PER_THREAD];
        {
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

        uint16_t offsets[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i) {
            WarpLevelMultisplit<RADIX_LOG, RADIX_MASK>(keys[i], radixShift, offsets[i], s_warpHist);
        }
        __syncthreads();

        //exclusive prefix sum up the warp histograms
        if (threadIdx.x < RADIX) {
            uint32_t reduction = s_warpHistograms[threadIdx.x];
            #pragma unroll
            for (uint32_t i = threadIdx.x + RADIX, k = 0; k < WARPS - 1; i += RADIX, ++k) {
                reduction += s_warpHistograms[i];
                s_warpHistograms[i] = reduction - s_warpHistograms[i];
            }

            if (partitionIndex < gridDim.x - 1) {
                atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
                          FLAG_REDUCTION | reduction << 2);
            }

            //begin the exclusive prefix sum across the reductions
            s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
        }
        __syncthreads();
        __threadfence();

        if (threadIdx.x < LANE_COUNT) {
            const uint32_t index = threadIdx.x * LANE_COUNT;
            const bool pred = index < RADIX;
            const uint32_t t = ExclusiveWarpScan(pred ? s_localHistogram[index] : 0);
            if (pred) {
                s_localHistogram[index] = t;
            }
        }
        __syncthreads();

        if (threadIdx.x < RADIX && getLaneId())
            s_localHistogram[threadIdx.x] +=
                __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
        __syncthreads();

        //update offsets
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
        __syncthreads();

        //scatter keys into shared memory
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k) {
            s_warpHistograms[offsets[k]] = keys[k];
        }

        //lookback
        if (threadIdx.x < RADIX) {
            uint32_t reduction = 0;
            uint32_t lookbackIndex = partitionIndex * RADIX;
            while (true) {
                const uint32_t flagPayload = passHistogram[threadIdx.x + lookbackIndex];
                if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) {
                    reduction += flagPayload >> 2;
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                        if (partitionIndex < gridDim.x - 1) {
                            atomicAdd((uint32_t*)&passHistogram[threadIdx.x +
                                                                (partitionIndex + 1) * RADIX],
                                      1 | (reduction << 2));
                        }
                        s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
                        break;
                    } else {
                        lookbackIndex -= RADIX;
                    }
                }
            }
        }
        __syncthreads();

        //Scatter runs of keys and/or values
        (*Scatter)(alt, payload, altPayload, offsets, s_warpHistograms, s_localHistogram,
                   radixShift, partitionIndex, size - partitionIndex * PART_SIZE);
    }

    template <uint32_t RADIX, uint32_t RADIX_LOG, uint32_t WARPS, uint32_t KEYS_PER_THREAD>
    __global__ void DigitBinningPassKeys(uint32_t* sort, uint32_t* alt,
                                         volatile uint32_t* passHistogram, volatile uint32_t* index,
                                         uint32_t size, uint32_t radixShift) {
        DigitBinningPass<RADIX, RADIX_LOG, WARPS, KEYS_PER_THREAD>(
            sort, sort, alt, alt, passHistogram, index, size, radixShift,
            ScatterKeysOnly<KEYS_PER_THREAD, RADIX - 1>);
    }

    template <uint32_t RADIX, uint32_t RADIX_LOG, uint32_t WARPS, uint32_t KEYS_PER_THREAD>
    __global__ void DigitBinningPassPairs(uint32_t* sort, uint32_t* payload, uint32_t* alt,
                                          uint32_t* altPayload, volatile uint32_t* passHistogram,
                                          volatile uint32_t* index, uint32_t size,
                                          uint32_t radixShift) {
        DigitBinningPass<RADIX, RADIX_LOG, WARPS, KEYS_PER_THREAD>(
            sort, payload, alt, altPayload, passHistogram, index, size, radixShift,
            ScatterPairs<WARPS, KEYS_PER_THREAD, RADIX - 1>);
    }
}  // namespace OneSweep

#undef FLAG_NOT_READY
#undef FLAG_REDUCTION
#undef FLAG_INCLUSIVE
#undef FLAG_MASK
