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
#include "SplitSortUtils.cuh"
#include "SplitSortBBUtils.cuh"

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0
#define FLAG_REDUCTION      1
#define FLAG_INCLUSIVE      2
#define FLAG_MASK           3

namespace SplitSortInternal
{
    struct BinInfo32
    {
        uint32_t binMask;
        uint32_t binOffset;
    };

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void MultiSplit32AsmGe(
        uint32_t& geMask,
        const uint32_t key)
    {
        #pragma unroll
        for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
        {
            uint32_t current_bit = 1 << bit;
            asm("{\n"
                "    .reg .pred p;\n"
                "    and.b32 %2, %1, %2;\n"
                "    setp.eq.u32 p, %2, 0;\n"
                "    vote.ballot.sync.b32 %2, p, 0xffffffff;\n"
                "    @p and.b32 %0, %0, %2;\n"
                "    @!p or.b32 %0, %0, %2;\n"
                "}\n" : "+r"(geMask) : "r"(key), "r"(current_bit));
        }
    }

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void CuteSort32BinGe(
        const uint32_t key,
        uint32_t& index,
        const BinInfo32 binInfo,
        const uint32_t totalLocalLength)
    {
        uint32_t geMask = getLaneMaskLt();
        MultiSplit32AsmGe<BITS_TO_SORT>(geMask, key);

        if (getLaneId() < totalLocalLength)
        {
            index = __popc(geMask & binInfo.binMask);
            index += binInfo.binOffset;
        }
    }

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void cs32Ge(
        uint32_t& key,
        uint2* s_pairs,
        const uint32_t totalLocalLength,
        const uint32_t runStart)
    {
        if (totalLocalLength - runStart > 16)
        {
            uint32_t geMask = getLaneMaskLt();
            MultiSplit32AsmGe<BITS_TO_SORT>(geMask, key);

            if (getLaneId() + runStart < totalLocalLength)
                s_pairs[__popc(geMask)] = { key, getLaneId() + runStart };
            else
                s_pairs[getLaneId()].x = 0xffffffff;
        }
        else
        {
            uint32_t index = getLaneId();
            RegSortFallback(key, index, totalLocalLength - runStart);
            if (getLaneId() + runStart < totalLocalLength)
                s_pairs[getLaneId()] = { key, index + runStart };
            else
                s_pairs[getLaneId()].x = 0xffffffff;
        }
    }

    template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void CuteSort32(
        uint32_t* keys,
        uint2* s_pairs,
        const uint32_t totalLocalLength,
        const uint32_t warpOffset)
    {
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
        {
            const uint32_t runStart = k * LANE_COUNT + warpOffset;
            if (runStart < totalLocalLength)
            {
                cs32Ge<BITS_TO_SORT>(
                    keys[k],
                    &s_pairs[k * LANE_COUNT],
                    totalLocalLength,
                    runStart);
            }
            else
            {
                s_pairs[getLaneId() + k * LANE_COUNT].x = 0xffffffff;
            }
        }
    }

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void MultiSplit64AsmGe(
        uint64_t& geMask0,
        uint64_t& geMask1,
        const uint32_t key0,
        const uint32_t key1)
    {
        #pragma unroll
        for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
        {
            uint32_t current_bit = 1 << bit;
            asm("{\n"
                "    .reg .pred p0;\n"
                "    .reg .pred p1;\n"
                "    .reg .b32 bal;\n"
                "    .reg .b64 t;\n"
                "    and.b32 bal, %2, %4;\n"
                "    setp.eq.u32 p0, bal, 0;\n"
                "    vote.ballot.sync.b32 bal, p0, 0xffffffff;\n"
                "    and.b32 %4, %3, %4;\n"
                "    setp.eq.u32 p1, %4, 0;\n"
                "    vote.ballot.sync.b32 %4, p1, 0xffffffff;\n"
                "    mov.b64 t, {bal, %4};\n"
                "    @p0 and.b64 %0, %0, t;\n"
                "    @p1 and.b64 %1, %1, t;\n"
                "    @!p0 or.b64 %0, %0, t;\n"
                "    @!p1 or.b64 %1, %1, t;\n"
                "}\n" : "+l"(geMask0), "+l"(geMask1) : "r"(key0), "r"(key1), "r"(current_bit));
        }
    }

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void cs64Ge(
        uint32_t& key0,
        uint32_t& key1,
        uint2* s_pairs,
        const uint32_t totalLocalLength,
        const uint32_t runStart)
    {
        if (totalLocalLength - runStart > 32)
        {
            uint64_t geMask0 = getLaneMaskLt();
            uint64_t geMask1 = geMask0 << 32 | 0xffffffff;
            MultiSplit64AsmGe<BITS_TO_SORT>(geMask0, geMask1, key0, key1);

            s_pairs[__popcll(geMask0)] = { key0, getLaneId() + runStart };

            if (getLaneId() + runStart + LANE_COUNT < totalLocalLength)
                s_pairs[__popcll(geMask1)] = { key1, getLaneId() + runStart + LANE_COUNT };
            else
                s_pairs[getLaneId() + LANE_COUNT].x = 0xffffffff;
        }
        else
        {
            cs32Ge<BITS_TO_SORT>(key0, s_pairs, totalLocalLength, runStart);
            s_pairs[getLaneId() + LANE_COUNT].x = 0xffffffff;
        }
    }

    template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void CuteSort64(
        uint32_t* keys,
        uint2* s_pairs,
        const uint32_t totalLocalLength,
        const uint32_t warpOffset)
    {
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; k += 2)
        {
            const uint32_t runStart = k * LANE_COUNT + warpOffset;
            if (runStart < totalLocalLength)
            {
                cs64Ge<BITS_TO_SORT>(
                    keys[k],
                    keys[k + 1],
                    &s_pairs[k >> 1 << 6],
                    totalLocalLength,
                    runStart);
            }
            else
            {
                s_pairs[getLaneId() + k * LANE_COUNT].x = 0xffffffff;
                s_pairs[getLaneId() + (k + 1) * LANE_COUNT].x = 0xffffffff;
            }
        }
    }

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void MultiSplit128AsmGe(
        uint64_t& geMask00, uint64_t& geMask01, uint64_t& geMask10, uint64_t& geMask11,
        uint64_t& geMask20, uint64_t& geMask21, uint64_t& geMask30, uint64_t& geMask31,
        const uint32_t key0, const uint32_t key1, const uint32_t key2, const uint32_t key3)
    {
        #pragma unroll
        for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
        {
            uint32_t current_bit = 1 << bit;
            asm("{\n"
                "    .reg .pred p0;\n"
                "    .reg .pred p1;\n"
                "    .reg .pred p2;\n"
                "    .reg .pred p3;\n"
                "    .reg .b32 bal0;\n"
                "    .reg .b32 bal1;\n"
                "    .reg .b32 bal2;\n"
                "    .reg .b64 t;\n"
                "    and.b32 bal0, %8, %12;\n"
                "    setp.eq.u32 p0, bal0, 0;\n"
                "    and.b32 bal1, %9, %12;\n"
                "    setp.eq.u32 p1, bal1, 0;\n"
                "    and.b32 bal2, %10, %12;\n"
                "    setp.eq.u32 p2, bal2, 0;\n"
                "    and.b32 %12, %11, %12;\n"
                "    setp.eq.u32 p3, %12, 0;\n"
                "    vote.ballot.sync.b32 bal0, p0, 0xffffffff;\n"
                "    vote.ballot.sync.b32 bal1, p1, 0xffffffff;\n"
                "    vote.ballot.sync.b32 bal2, p2, 0xffffffff;\n"
                "    vote.ballot.sync.b32 %12, p3, 0xffffffff;\n"
                "    mov.b64 t, {bal0, bal1};\n"
                "    @p0 and.b64 %0, %0, t;\n"
                "    @p1 and.b64 %2, %2, t;\n"
                "    @p2 and.b64 %4, %4, t;\n"
                "    @p3 and.b64 %6, %6, t;\n"
                "    @!p0 or.b64 %0, %0, t;\n"
                "    @!p1 or.b64 %2, %2, t;\n"
                "    @!p2 or.b64 %4, %4, t;\n"
                "    @!p3 or.b64 %6, %6, t;\n"
                "    mov.b64 t, {bal2, %12};\n"
                "    @p0 and.b64 %1, %1, t;\n"
                "    @p1 and.b64 %3, %3, t;\n"
                "    @p2 and.b64 %5, %5, t;\n"
                "    @p3 and.b64 %7, %7, t;\n"
                "    @!p0 or.b64 %1, %1, t;\n"
                "    @!p1 or.b64 %3, %3, t;\n"
                "    @!p2 or.b64 %5, %5, t;\n"
                "    @!p3 or.b64 %7, %7, t;\n"
                "}\n" : "+l"(geMask00), "+l"(geMask01), "+l"(geMask10), "+l"(geMask11),
                "+l"(geMask20), "+l"(geMask21), "+l"(geMask30), "+l"(geMask31) :
                "r"(key0), "r"(key1), "r"(key2), "r"(key3), "r"(current_bit));
        }
    }

    template<uint32_t BITS_TO_SORT>
    __device__ __forceinline__ void cs128Ge(
        uint32_t& key0, uint32_t& key1, uint32_t& key2, uint32_t& key3,
        uint2* s_pairs,
        const uint32_t totalLocalLength,
        const uint32_t runStart)
    {
        if (totalLocalLength - runStart > 64)
        {
            uint64_t geMask00 = getLaneMaskLt();
            uint64_t geMask01 = 0;
            uint64_t geMask10 = geMask00 << 32ULL | 0xffffffff;
            uint64_t geMask11 = 0;
            uint64_t geMask20 = 0xffffffffffffffff;
            uint64_t geMask21 = getLaneMaskLt();
            uint64_t geMask30 = 0xffffffffffffffff;
            uint64_t geMask31 = geMask10;

            MultiSplit128AsmGe<BITS_TO_SORT>(
                geMask00, geMask01, geMask10, geMask11,
                geMask20, geMask21, geMask30, geMask31,
                key0, key1, key2, key3);

            s_pairs[__popcll(geMask00) + __popcll(geMask01)] = { key0, getLaneId() + runStart };
            s_pairs[__popcll(geMask10) + __popcll(geMask11)] = { key1, getLaneId() + runStart + 32 };

            if (getLaneId() + runStart + 64 < totalLocalLength)
                s_pairs[__popcll(geMask20) + __popcll(geMask21)] = { key2, getLaneId() + runStart + 64 };
            else
                s_pairs[getLaneId() + 64].x = 0xffffffff;

            if (getLaneId() + runStart + 96 < totalLocalLength)
                s_pairs[__popcll(geMask30) + __popcll(geMask31)] = { key3, getLaneId() + runStart + 96 };
            else
                s_pairs[getLaneId() + 96].x = 0xffffffff;
        }
        else
        {
            if (totalLocalLength - runStart > 32)
            {
                cs64Ge<BITS_TO_SORT>(
                    key0,
                    key1,
                    s_pairs,
                    totalLocalLength,
                    runStart);
                s_pairs[getLaneId() + 64].x = 0xffffffff;
                s_pairs[getLaneId() + 96].x = 0xffffffff;
            }
            else
            {
                cs32Ge<BITS_TO_SORT>(
                    key0,
                    s_pairs,
                    totalLocalLength,
                    runStart);
                s_pairs[getLaneId() + 32].x = 0xffffffff;
                s_pairs[getLaneId() + 64].x = 0xffffffff;
                s_pairs[getLaneId() + 96].x = 0xffffffff;
            }
        }
    }

    //Sort 128 keys at a time instead of 32, KEYS_PER_THREAD must be a multiple of 4
    template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
    __device__ __forceinline__ void CuteSort128(
        uint32_t* keys,
        uint2* s_pairs,
        const uint32_t totalLocalLength,
        const uint32_t warpOffset)
    {
        #pragma unroll
        for (uint32_t k = 0; k < KEYS_PER_THREAD; k += 4)
        {
            const uint32_t runStart = k * LANE_COUNT + warpOffset;
            if (runStart < totalLocalLength)
            {
                cs128Ge<BITS_TO_SORT>(
                    keys[k], keys[k + 1], keys[k + 2], keys[k + 3],
                    &s_pairs[k >> 2 << 7],
                    totalLocalLength,
                    runStart);
            }
            else
            {
                s_pairs[getLaneId() + k * LANE_COUNT].x = 0xffffffff;
                s_pairs[getLaneId() + (k + 1) * LANE_COUNT].x = 0xffffffff;
                s_pairs[getLaneId() + (k + 2) * LANE_COUNT].x = 0xffffffff;
                s_pairs[getLaneId() + (k + 3) * LANE_COUNT].x = 0xffffffff;
            }
        }
    }

    __device__ __forceinline__ void LoadBins(
        const uint32_t* segments,
        uint32_t* s_warpBins,
        const uint32_t packSegCount,
        const uint32_t binOffset,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        s_warpBins[getLaneId()] = getLaneId() + binOffset >= totalSegCount ?
            totalSegLength : segments[getLaneId() + binOffset];
        __syncwarp(0xffffffff);
    }

    __device__ __forceinline__ BinInfo32 GetBinInfo32(
        const uint32_t* s_warpBins,
        const uint32_t packSegCount)
    {
        const uint2 interval = BinarySearch(s_warpBins, (int32_t)packSegCount, getLaneId());
        const uint32_t binMask = (((1ULL << interval.y) - 1) >> interval.x << interval.x);
        return BinInfo32{ binMask, interval.x };
    }

    __device__ __forceinline__ void SingleBinFallback(
        uint32_t& key,
        uint32_t& index,
        const uint32_t totalLocalLength)
    {
        index = getLaneId();
        RegSortFallback(key, index, totalLocalLength);
    }

    // Only a single warp participates in sorting a run of keys
    // However, multiple independent warps can be launched in a single block
    template<
        uint32_t WARP_KEYS,
        uint32_t BLOCK_KEYS,
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class V>
    __device__ __forceinline__ void SplitSortBins32(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* packedSegCounts,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        if (blockIdx.x * WARPS + WARP_INDEX >= segCountInBin)
            return;

        __shared__ uint32_t s_bins[LANE_COUNT * WARPS];
        uint32_t* s_warpBins = &s_bins[WARP_INDEX * LANE_COUNT];

        const uint32_t binOffset = binOffsets[blockIdx.x * WARPS + WARP_INDEX];
        const uint32_t packSegCount = packedSegCounts[blockIdx.x * WARPS + WARP_INDEX];

        //If the packSegCount count is 32, then all bins must be of size 1, so we short circuit
        if (packSegCount == 32) //TODO this will not work with bins of 0 
            return;

        LoadBins(segments, s_warpBins, packSegCount, binOffset, totalSegCount, totalSegLength);
        const uint32_t segmentStart = s_warpBins[0];
        const uint32_t totalLocalLength = s_warpBins[packSegCount] - segmentStart;
        sort += segmentStart;
        values += segmentStart;

        uint32_t key = getLaneId() < totalLocalLength ? sort[getLaneId()] : 0xffffffff;

        //If the packSegCount is 1, and the length of the segment
        //is short, skip cute sort and use a regSort style fallback
        uint32_t index;
        V val;
        if (packSegCount == 1 && totalLocalLength <= 16)
        {
            SingleBinFallback(key, index, totalLocalLength);
            if (getLaneId() < totalLocalLength)
                val = values[index];
            __syncwarp(0xffffffff);
            if (getLaneId() < totalLocalLength)
            {
                sort[getLaneId()] = key;
                values[getLaneId()] = val;
            }
        }
        else
        {
            CuteSort32BinGe<BITS_TO_SORT>(key, index, GetBinInfo32(s_warpBins, packSegCount), totalLocalLength);
            if (getLaneId() < totalLocalLength)
                val = values[getLaneId()];
            __syncwarp(0xffffffff);
            if (getLaneId() < totalLocalLength)
            {
                sort[index] = key;
                values[index] = val;
            }
        }
    }

    __device__ __forceinline__ void MergeGather(
        const uint2* source,
        uint2* dest,
        uint32_t startA,
        uint32_t startB,
        const uint32_t mergeLength,
        const uint32_t tMergeLength)
    {
        for (uint32_t i = 0; i < tMergeLength; ++i)
        {
            const uint2 t0 = startA < mergeLength ? source[startA] : uint2{ 0xffffffff, 0xffffffff };
            const uint2 t1 = startB < (mergeLength << 1) ? source[startB] : uint2{ 0xffffffff, 0xffffffff };
            bool pred = startB >= (mergeLength << 1) || (startA < mergeLength&& t0.x <= t1.x);

            if (pred)
            {
                dest[i] = t0;
                ++startA;
            }
            else
            {
                dest[i] = t1;
                ++startB;
            }
        }
    }

    __device__ __forceinline__ void MergeScatter(
        const uint32_t id,
        uint2* s_pairs,
        const uint2* pairs,
        const uint32_t stride,
        const uint32_t totalLocalLength)
    {
        const uint32_t start = id * stride;
        if (start < totalLocalLength)
        {
            #pragma unroll
            for (uint32_t i = start, k = 0; k < stride; ++i, ++k)
                s_pairs[i] = pairs[k];
        }
    }

    __device__ __forceinline__ void Merge(
        const uint2* s_pairs,
        uint2* pairs,
        const uint32_t mergeId,
        const uint32_t mergeThreads,
        const uint32_t mergeLength,
        const uint32_t totalLocalLength)
    {
        const uint32_t tStart = (mergeId << 1) * mergeLength / mergeThreads;
        if (tStart < totalLocalLength)
        {
            const uint32_t startA = find_kth3(
                s_pairs,
                &s_pairs[mergeLength],
                mergeLength,
                tStart);
            const uint32_t startB = mergeLength + tStart - startA;

            MergeGather(
                s_pairs,
                pairs,
                startA,
                startB,
                mergeLength,
                (mergeLength << 1) / mergeThreads);
        }
    }

    template<
        uint32_t START_LOG,
        uint32_t END_LOG,
        uint32_t KEYS_PER_THREAD,
        bool SHOULD_SCATTER_FINAL>
    __device__ __forceinline__ void MultiLevelMergeWarp(
        uint2* s_pairs,
        uint2* pairs,
        const uint32_t totalLocalLength)
    {
        #pragma unroll
        for (uint32_t m = START_LOG; m < END_LOG; ++m)
        {
            #pragma unroll
            for (uint32_t i = 0; i < (1 << END_LOG - m); i += 2)
            {
                const uint32_t mergeStart = i << m;
                Merge(
                    &s_pairs[mergeStart],
                    &pairs[mergeStart >> LANE_LOG],
                    getLaneId(),
                    LANE_COUNT,
                    1 << m,
                    totalLocalLength);
            }
            __syncwarp(0xffffffff);

            if (m < END_LOG - 1 || SHOULD_SCATTER_FINAL)
            {
                #pragma unroll
                for (uint32_t i = 0; i < (1 << END_LOG - m); i += 2)
                {
                    const uint32_t mergeStart = i << m;
                    const uint32_t regStart = mergeStart >> LANE_LOG;
                    MergeScatter(
                        getLaneId(),
                        &s_pairs[mergeStart],
                        &pairs[regStart],
                        1 << m - LANE_LOG + 1,
                        totalLocalLength);
                }
            }
            __syncwarp(0xffffffff);
        }
    }

    template<
        uint32_t KEYS_PER_THREAD,
        uint32_t WARP_KEYS,
        uint32_t BLOCK_KEYS,
        uint32_t WARPS,
        uint32_t WARP_LOG_START,
        uint32_t WARP_LOG_END,
        class V>
    __device__ __forceinline__ void SplitSortWarp(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin,
        void (*CuteSortVariant)(uint32_t*, uint2*, const uint32_t, const uint32_t))
    {
        if (blockIdx.x * WARPS + WARP_INDEX >= segCountInBin)
            return;

        __shared__ uint2 s_mem[BLOCK_KEYS];
        uint2* s_warpPairs = &s_mem[WARP_INDEX * WARP_KEYS];

        const uint32_t binOffset = binOffsets[blockIdx.x * WARPS + WARP_INDEX];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        values += segmentStart;

        uint32_t keys[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = getLaneId(), k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            keys[k] = i < totalLocalLength ? sort[i] : 0xffffffff;
        }

        (*CuteSortVariant)(keys, s_warpPairs, totalLocalLength, 0);
        __syncwarp(0xffffffff);

        uint2 pairs[KEYS_PER_THREAD];
        MultiLevelMergeWarp<
            WARP_LOG_START,
            WARP_LOG_END,
            KEYS_PER_THREAD,
            false>(
                s_warpPairs,
                pairs,
                totalLocalLength);

        //If no merging was needed, scatter straight from shared memory
        V vals[KEYS_PER_THREAD];
        if constexpr (WARP_LOG_END == WARP_LOG_START)
        {
            #pragma unroll
            for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                    sort[i] = s_warpPairs[i].x;
            }

            #pragma unroll
            for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                    vals[k] = values[s_warpPairs[i].y];
            }
            __syncwarp(0xffffffff);

            #pragma unroll
            for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                    values[i] = vals[k];
            }
        }

        //Else, scatter the post merge results from registers, for most
        //warp sized partition workloads, prescattering to shared memory is a slowdown
        if constexpr (WARP_LOG_END > WARP_LOG_START)
        {
            #pragma unroll
            for (uint32_t i = getLaneId() * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
            {
                if (i < totalLocalLength)
                    sort[i] = pairs[k].x;
            }

            #pragma unroll
            for (uint32_t i = getLaneId() * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
            {
                if (i < totalLocalLength)
                    vals[k] = values[pairs[k].y];
            }
            __syncwarp(0xffffffff);

            #pragma unroll
            for (uint32_t i = getLaneId() * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
            {
                if (i < totalLocalLength)
                    values[i] = vals[k];
            }
        }
    }

    //number of participating warps is implied by the difference between starting and ending log
    template<
        uint32_t START_LOG,
        uint32_t END_LOG,
        uint32_t KEYS_PER_THREAD,
        bool SHOULD_SCATTER_FINAL>
    __device__ __forceinline__ void MultiLevelMergeBlock(
        uint2* s_pairs,
        uint2* pairs,
        const uint32_t totalLocalLength)
    {
        #pragma unroll
        for (uint32_t m = START_LOG, w = 1; m < END_LOG; ++m, ++w)
        {
            const uint32_t mergeStart = WARP_INDEX >> w << m + 1;
            const uint32_t mergeLength = 1 << m;
            const uint32_t mergeThreads = 1 << w << LANE_LOG;
            const uint32_t mergeId = threadIdx.x & mergeThreads - 1;

            Merge(
                &s_pairs[mergeStart],
                pairs,
                mergeId,
                mergeThreads,
                mergeLength,
                totalLocalLength);
            __syncthreads();

            if (m < END_LOG - 1 || SHOULD_SCATTER_FINAL)
            {
                MergeScatter(
                    mergeId,
                    &s_pairs[mergeStart],
                    pairs,
                    KEYS_PER_THREAD,
                    totalLocalLength);
            }
            __syncthreads();
        }
    }

    template<
        uint32_t KEYS_PER_THREAD,
        uint32_t WARP_KEYS,
        uint32_t BLOCK_KEYS,
        uint32_t WARPS,
        uint32_t WARP_LOG_START,
        uint32_t WARP_LOG_END,
        uint32_t BLOCK_LOG_END,
        bool SHOULD_PRE_SCATTER,
        class V>
    __device__ __forceinline__ void SplitSortBlock(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        void (*CuteSortVariant)(uint32_t*, uint2*, const uint32_t, const uint32_t))
    {
        __shared__ uint2 s_blockPairs[BLOCK_KEYS];
        uint2* s_warpPairs = &s_blockPairs[WARP_INDEX * WARP_KEYS];

        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        values += segmentStart;

        uint32_t keys[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * WARP_KEYS, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            keys[k] = i < totalLocalLength ? sort[i] : 0xffffffff;
        }

        (*CuteSortVariant)(keys, s_warpPairs, totalLocalLength, WARP_INDEX * WARP_KEYS);
        __syncwarp(0xffffffff);

        uint2 pairs[KEYS_PER_THREAD];
        MultiLevelMergeWarp<
            WARP_LOG_START,
            WARP_LOG_END,
            KEYS_PER_THREAD,
            true>(
                s_warpPairs,
                pairs,
                totalLocalLength);
        __syncthreads();

        MultiLevelMergeBlock<
            WARP_LOG_END,
            BLOCK_LOG_END,
            KEYS_PER_THREAD,
            SHOULD_PRE_SCATTER>(
                s_blockPairs,
                pairs,
                totalLocalLength);

        V vals[KEYS_PER_THREAD];
        if constexpr (SHOULD_PRE_SCATTER)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (i < totalLocalLength)
                    sort[i] = s_blockPairs[i].x;
            }

            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (i < totalLocalLength)
                    vals[k] = values[s_blockPairs[i].y];
            }
            __syncthreads();

            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (i < totalLocalLength)
                    values[i] = vals[k];
            }
        }

        if constexpr (!SHOULD_PRE_SCATTER)
        {
            #pragma unroll
            for (uint32_t i = threadIdx.x * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
            {
                if (i < totalLocalLength)
                    sort[i] = pairs[k].x;
            }

            #pragma unroll
            for (uint32_t i = threadIdx.x * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
            {
                if (i < totalLocalLength)
                    vals[k] = values[pairs[k].y];
            }
            __syncthreads();

            #pragma unroll
            for (uint32_t i = threadIdx.x * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
            {
                if (i < totalLocalLength)
                    values[i] = vals[k];
            }
        }
    }

    //TODO remove uncessary ternary
    template<uint32_t BITS_TO_RANK>
    __device__ __forceinline__ void MultiSplitRadixAsm(
        uint32_t& eqMask,
        const uint32_t key,
        const uint32_t radixShift)
    {
        eqMask = 0xffffffff;
        #pragma unroll
        for (uint32_t bit = 0; bit < BITS_TO_RANK; ++bit)
        {
            uint32_t current_bit = 1 << bit + radixShift;
            asm("{\n"
                "    .reg .pred p;\n"
                "    .reg .b32 bal;\n"
                "    and.b32 bal, %1, %2;\n"
                "    setp.ne.u32 p, bal, 0;\n"
                "    vote.ballot.sync.b32 bal, p, 0xffffffff;\n"
                "    @!p not.b32 bal, bal;\n"
                "    and.b32 %0, %0, bal;\n"
                "}\n" : "+r"(eqMask) : "r"(key), "r"(current_bit));
        }
    }

    template<
        uint32_t KEYS_PER_THREAD,
        uint32_t BITS_TO_RANK,
        uint32_t MASK>
        __device__ __forceinline__ void RankKeys(
            uint32_t* keys,
            uint32_t* offsets,
            uint32_t* warpHist,
            const uint32_t radixShift)
    {
        #pragma unroll
        for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
        {
            uint32_t eqMask;
            MultiSplitRadixAsm<BITS_TO_RANK>(eqMask, keys[i], radixShift);
            const uint32_t ltEqPeers = __popc(eqMask & getLaneMaskLt());
            uint32_t preIncrementVal;
            if (ltEqPeers == 0)
                preIncrementVal = atomicAdd((uint32_t*)&warpHist[keys[i] >> radixShift & MASK], __popc(eqMask));
            offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(eqMask) - 1) + ltEqPeers;
        }
    }

    template<uint32_t HIST_SIZE>
    __device__ __forceinline__ void ClearWarpHist(
        uint32_t* s_warpHist)
    {
        for (uint32_t i = getLaneId(); i < HIST_SIZE; i += LANE_COUNT)
            s_warpHist[i] = 0;
    }

    //Get the totalLocalLength of a segment and advance
    //the device pointers to the correction location
    template<class V>
    __device__ __forceinline__ void GetSegmentInfoRadixFine(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t*& sort,
        V*& values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        uint32_t& totalLocalLength)
    {
        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        values += segmentStart;
    }

    template<
        uint32_t WARPS,
        uint32_t KEYS_PER_THREAD,
        uint32_t KEYS_PER_WARP,
        uint32_t PART_SIZE,
        uint32_t BITS_TO_SORT,
        uint32_t RADIX,
        uint32_t RADIX_MASK,
        uint32_t RADIX_LOG,
        class V>
    __device__ __forceinline__ void SplitSortRadixFine(
        uint32_t* s_hist,
        uint32_t* s_indexes,
        uint32_t* sort,
        V* values,
        const uint32_t totalLocalLength)
    {
        uint32_t* s_warpHist = &s_hist[WARP_INDEX * RADIX];
        ClearWarpHist<RADIX>(s_warpHist);

        uint32_t keys[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            keys[k] = i < totalLocalLength ? sort[i] : 0xffffffff;
        }
        __syncthreads();

        uint32_t offsets[KEYS_PER_THREAD];
        uint32_t indexes[KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t radixShift = 0; radixShift < BITS_TO_SORT; radixShift += RADIX_LOG)
        {
            if (radixShift)
            {
                ClearWarpHist<RADIX>(s_warpHist);
                __syncthreads();
            }

            RankKeys<KEYS_PER_THREAD, RADIX_LOG, RADIX_MASK>(
                keys,
                offsets,
                s_warpHist,
                radixShift);
            __syncthreads();

            for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
            {
                uint32_t reduction = s_hist[i];
                for (uint32_t k = i + RADIX; k < (RADIX * WARPS); k += RADIX)
                {
                    reduction += s_hist[k];
                    s_hist[k] = reduction - s_hist[k];
                }

                s_hist[i] = InclusiveWarpScanCircularShift(reduction);
            }
            __syncthreads();

            if (threadIdx.x < LANE_COUNT)
            {
                const bool p = threadIdx.x < (RADIX >> LANE_LOG);
                const uint32_t t = ExclusiveWarpScan(p ? s_hist[threadIdx.x << LANE_LOG] : 0);
                if(p)
                    s_hist[threadIdx.x << LANE_LOG] = t;
            }
            __syncthreads();

            for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
            {
                if (getLaneId())
                    s_hist[i] += __shfl_sync(0xfffffffe, s_hist[i - 1], 1);
            }
            __syncthreads();

            if (threadIdx.x >= LANE_COUNT)
            {
                #pragma unroll
                for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
                {
                    const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
                    offsets[i] += s_warpHist[t2] + s_hist[t2];
                }
            }
            else
            {
                #pragma unroll
                for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
                    offsets[i] += s_hist[keys[i] >> radixShift & RADIX_MASK];
            }
            __syncthreads();

            if (radixShift)
            {
                #pragma unroll
                for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                    k < KEYS_PER_THREAD;
                    i += LANE_COUNT, ++k)
                {
                    s_hist[offsets[k]] = keys[k];
                    s_indexes[i] = offsets[k];
                }
            }
            else
            {
                #pragma unroll
                for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                    k < KEYS_PER_THREAD;
                    i += LANE_COUNT, ++k)
                {
                    s_hist[offsets[k]] = keys[k];
                    indexes[k] = offsets[k];
                }
            }
            __syncthreads();

            if (radixShift < BITS_TO_SORT - RADIX_LOG)
            {
                if (radixShift)
                {
                    #pragma unroll
                    for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                        k < KEYS_PER_THREAD;
                        i += LANE_COUNT, ++k)
                    {
                        keys[k] = s_hist[i];
                        indexes[k] = s_indexes[indexes[k]];
                    }
                }
                else
                {
                    #pragma unroll
                    for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                        k < KEYS_PER_THREAD;
                        i += LANE_COUNT, ++k)
                    {
                        keys[k] = s_hist[i];
                    }
                }
                __syncthreads();
            }
        }

        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if (i < totalLocalLength)
                sort[i] = s_hist[i];
        }

        //If possible, scatter the values into shared memory prior to device
        V vals[KEYS_PER_THREAD];
        if constexpr (sizeof(V) * PART_SIZE <= (RADIX * WARPS + PART_SIZE) * sizeof(uint32_t))
        {
            #pragma unroll
            for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                    vals[k] = values[i];
            }

            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
                indexes[k] = s_indexes[indexes[k]];
            __syncthreads();

            V* s_payloadsOut = reinterpret_cast<V*>(s_hist);
            #pragma unroll
            for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                s_payloadsOut[indexes[k]] = vals[k];
            }
            __syncthreads();

            #pragma unroll
            for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
            {
                if (i < totalLocalLength)
                    values[i] = s_payloadsOut[i];
            }
        }

        if constexpr (sizeof(V) * PART_SIZE > (RADIX * WARPS + PART_SIZE) * sizeof(uint32_t))
        {
            #pragma unroll
            for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                    vals[k] = values[i];
            }
            __syncthreads();

            #pragma unroll
            for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                    values[s_indexes[indexes[k]]] = vals[k];
            }
        }
    }
}

#undef FLAG_MASK
#undef FLAG_INCLUSIVE
#undef FLAG_REDUCTION
#undef FLAG_NOT_READY