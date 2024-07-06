/******************************************************************************
 * GPUSorting
 * SplitSort
 * Experimental Hybrid Radix-Merge based SegmentedSort
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 5/16/2024
 * https://github.com/b0nes164/GPUSorting
 *
 * Improving on "CuteSort" technique originally suggested by
 *          Dondragmer
 *          https://gist.github.com/dondragmer/0c0b3eed0f7c30f7391deb11121a5aa1
 * 
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Utils.cuh"
#include "BBUtils.cuh"

struct BinInfo32
{
    uint32_t binMask;
    uint32_t binOffset;
};

struct BinInfo64
{
    uint64_t binMask;
    uint32_t binOffset;
};

__device__ __forceinline__ uint32_t ExtractDigit(const uint32_t key, const uint32_t bit)
{
    return key >> bit & 1;
}

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void MultiSplit32(
    uint32_t& eqMask,
    uint32_t& gtMask,
    const uint32_t key)
{
    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
    {
        const bool isBitSet = ExtractDigit(key, bit);
        const unsigned ballot = __ballot_sync(0xffffffff, isBitSet);

        if (isBitSet)
        {
            eqMask &= ballot;
            gtMask |= ~ballot;
        }
        else
        {
            eqMask &= ~ballot;
            gtMask &= ~ballot;
        }
    }
}

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void MultiSplit32Asm(
    uint32_t& eqMask,
    uint32_t& gtMask,
    const uint32_t key)
{
    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
    {
        uint32_t current_bit = 1 << bit;
        asm("{\n"
            "    .reg .pred p;\n"
            "    and.b32 %3, %2, %3;\n"
            "    setp.ne.u32 p, %3, 0;\n"
            "    vote.ballot.sync.b32 %3, p, 0xffffffff;\n"
            "    @p and.b32 %0, %0, %3;\n"
            "    not.b32 %3, %3;\n"
            "    @p or.b32 %1, %1, %3;\n"
            "    @!p and.b32 %1, %1, %3;\n"
            "    @!p and.b32 %0, %0, %3;\n"
            "}\n" : "+r"(eqMask), "+r"(gtMask) : "r"(key), "r"(current_bit));
    }
}

//Best
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

        /*bool b;
        current_bit = key & current_bit;
        b = current_bit == 0;
        current_bit = __ballot_sync(0xffffffff, b);
        if (b)
            geMask &= current_bit;
        else
            geMask |= current_bit;*/
    }
}

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void CuteSort32Bin(
    const uint32_t key,
    uint32_t& index,
    const BinInfo32 binInfo,
    const uint32_t totalLocalLength)
{
    uint32_t eqMask = 0xffffffff;
    uint32_t gtMask = 0;
    MultiSplit32Asm<BITS_TO_SORT>(eqMask, gtMask, key);

    if (getLaneId() < totalLocalLength)
    {
        index = __popc(gtMask & binInfo.binMask);
        index += __popc(eqMask & binInfo.binMask & getLaneMaskLt());
        index += binInfo.binOffset;
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
__device__ __forceinline__ void cs32(
    uint32_t& key,
    uint2* s_pairs,
    const uint32_t totalLocalLength,
    const uint32_t runStart)
{
    if (totalLocalLength - runStart > 16)
    {
        uint32_t eqMask = 0xffffffff;
        uint32_t gtMask = 0;
        MultiSplit32Asm<BITS_TO_SORT>(eqMask, gtMask, key);

        if (getLaneId() + runStart < totalLocalLength)
        {
            const uint32_t t = __popc(gtMask) + __popc(eqMask & getLaneMaskLt());
            s_pairs[t] = { key, getLaneId() + runStart };
        }
        else
        {
            s_pairs[getLaneId()].x = 0xffffffff;
        }
    }
    else
    {
        uint32_t index = getLaneId();
        BBUtils::RegSortFallback(key, index, totalLocalLength - runStart);
        if (getLaneId() + runStart < totalLocalLength)
            s_pairs[getLaneId()] = { key, index + runStart };
        else
            s_pairs[getLaneId()].x = 0xffffffff;
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
        BBUtils::RegSortFallback(key, index, totalLocalLength - runStart);
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
__device__ __forceinline__ void MultiSplit64(
    uint32_t& eqMask0,
    uint64_t& eqMask1,
    uint64_t& gtMask0,
    uint64_t& gtMask1,
    const uint32_t key0,
    const uint32_t key1)
{
    uint32_t ballot[2];
    bool setBit[2];

    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
    {
        setBit[0] = ExtractDigit(key0, bit);
        ballot[0] = __ballot_sync(0xffffffff, setBit[0]);

        setBit[1] = ExtractDigit(key1, bit);
        ballot[1] = __ballot_sync(0xffffffff, setBit[1]);

        const uint64_t t = ~reinterpret_cast<uint64_t*>(ballot)[0];
        if (setBit[0])
        {
            eqMask0 &= ballot[0];
            gtMask0 |= t;
        }
        else
        {
            eqMask0 &= (uint32_t)t;
            gtMask0 &= t;
        }

        if (setBit[1])
        {
            eqMask1 &= reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask1 |= t;
        }
        else
        {
            eqMask1 &= t;
            gtMask1 &= t;
        }
    }
}

//Asm 25% faster
template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void MultiSplit64Asm(
    uint32_t& eqMask0,
    uint64_t& eqMask1,
    uint64_t& gtMask0,
    uint64_t& gtMask1,
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
            "    .reg .b64 inv;\n"
            "    and.b32 bal, %4, %6;\n"
            "    setp.ne.u32 p0, bal, 0;\n"
            "    vote.ballot.sync.b32 bal, p0, 0xffffffff;\n"
            "    and.b32 %6, %5, %6;\n"
            "    setp.ne.u32 p1, %6, 0;\n"
            "    vote.ballot.sync.b32 %6, p1, 0xffffffff;\n"
            "    mov.b64 inv, {bal, %6};\n"
            "    @p1 and.b64 %1, %1, inv;\n"
            "    not.b64 inv, inv;\n"
            "    @p0 and.b32 %0, %0, bal;\n"
            "    @p0 or.b64 %2, %2, inv;\n"
            "    @!p0 and.b64 %2, %2, inv;\n"
            "    @p1 or.b64 %3, %3, inv;\n"
            "    @!p1 and.b64 %1, %1, inv;\n"
            "    @!p1 and.b64 %3, %3, inv;\n"
            "    cvt.u32.u64 bal, inv;\n"
            "    @!p0 and.b32 %0, %0, bal;\n"
            "}\n" : "+r"(eqMask0), "+l"(eqMask1), "+l"(gtMask0), "+l"(gtMask1) :
            "r"(key0), "r"(key1), "r"(current_bit));
    }
}

//Hack for unique keys only
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

//Sort 64 keys at a time instead of 32, KEYS_PER_THREAD must be even
template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void CuteSort64Bin(
    uint32_t* keys,
    uint32_t* indexes,
    const BinInfo64 binInfo0,
    const BinInfo64 binInfo1,
    const uint32_t totalLocalLength)
{
    uint32_t eqMask0 = ~0U;
    uint64_t eqMask1 = ~0ULL;
    uint64_t gtMask0 = 0;
    uint64_t gtMask1 = 0;
    MultiSplit64Asm<BITS_TO_SORT>(eqMask0, eqMask1, gtMask0, gtMask1, keys[0], keys[1]);

    if (getLaneId() < totalLocalLength)
    {
        indexes[0] = __popcll(gtMask0 & binInfo0.binMask);
        indexes[0] += __popc(eqMask0 & getLaneMaskLt() & (uint32_t)binInfo0.binMask);
        indexes[0] += binInfo0.binOffset;
    }

    if (getLaneId() + LANE_COUNT < totalLocalLength)
    {
        indexes[1] = __popcll(gtMask1 & binInfo1.binMask);
        indexes[1] += __popcll(eqMask1 & ((uint64_t)getLaneMaskLt() << 32 | 0xffffffff) & binInfo1.binMask);
        indexes[1] += binInfo1.binOffset;
    }
}

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void cs64(
    uint32_t& key0,
    uint32_t& key1,
    uint2* s_pairs,
    const uint32_t totalLocalLength,
    const uint32_t runStart)
{
    if (totalLocalLength - runStart > 32)
    {
        uint32_t eqMask0 = ~0U;
        uint64_t eqMask1 = ~0ULL;
        uint64_t gtMask0 = 0;
        uint64_t gtMask1 = 0;
        MultiSplit64Asm<BITS_TO_SORT>(eqMask0, eqMask1, gtMask0, gtMask1, key0, key1);

        {
            uint32_t t = __popc(eqMask0 & getLaneMaskLt());
            t += __popcll(gtMask0);
            s_pairs[t] = { key0, getLaneId() + runStart };
        }

        if (getLaneId() + runStart + LANE_COUNT < totalLocalLength)
        {
            uint32_t t = __popcll(eqMask1 & ((uint64_t)getLaneMaskLt() << 32 | 0xffffffff));
            t += __popcll(gtMask1);
            s_pairs[t] = { key1, getLaneId() + runStart + LANE_COUNT };
        }
        else
        {
            s_pairs[getLaneId() + LANE_COUNT].x = 0xffffffff;
        }
    }
    else
    {
        cs32<BITS_TO_SORT>(key0, s_pairs, totalLocalLength, runStart);
        s_pairs[getLaneId() + LANE_COUNT].x = 0xffffffff;
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
        uint64_t geMask1 = (uint64_t)getLaneMaskLt() << 32 | 0xffffffff;
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
__device__ __forceinline__ void MultiSplit128(
    uint32_t& eqMask00, uint64_t& eqMask10, uint64_t& eqMask20, uint32_t& eqMask21,
    uint64_t& eqMask30, uint64_t& eqMask31, uint64_t& gtMask00, uint64_t& gtMask01,
    uint64_t& gtMask10, uint64_t& gtMask11, uint64_t& gtMask20, uint64_t& gtMask21,
    uint64_t& gtMask30,  uint64_t& gtMask31,
    const uint32_t key0, const uint32_t key1, const uint32_t key2, const uint32_t key3)
{
    uint32_t ballot[4];
    bool setBit[4];

    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
    {
        setBit[0] = ExtractDigit(key0, bit);
        ballot[0] = __ballot_sync(0xffffffff, setBit[0]);
        setBit[1] = ExtractDigit(key1, bit);
        ballot[1] = __ballot_sync(0xffffffff, setBit[1]);
        setBit[2] = ExtractDigit(key2, bit);
        ballot[2] = __ballot_sync(0xffffffff, setBit[2]);
        setBit[3] = ExtractDigit(key3, bit);
        ballot[3] = __ballot_sync(0xffffffff, setBit[3]);

        if (setBit[0])
        {
            eqMask00 &= ballot[0];
            gtMask00 |= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask01 |= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }
        else
        {
            eqMask00 &= ~ballot[0];
            gtMask00 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask01 &= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }

        if (setBit[1])
        {
            eqMask10 &= reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask10 |= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask11 |= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }
        else
        {
            eqMask10 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask10 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask11 &= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }

        if (setBit[2])
        {
            eqMask20 &= reinterpret_cast<uint64_t*>(ballot)[0];
            eqMask21 &= ballot[2];
            gtMask20 |= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask21 |= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }
        else
        {
            eqMask20 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            eqMask21 &= ~ballot[2];
            gtMask20 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask21 &= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }

        if (setBit[3])
        {
            eqMask30 &= reinterpret_cast<uint64_t*>(ballot)[0];
            eqMask31 &= reinterpret_cast<uint64_t*>(ballot)[1];
            gtMask30 |= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask31 |= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }
        else
        {
            eqMask30 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            eqMask31 &= ~reinterpret_cast<uint64_t*>(ballot)[1];
            gtMask30 &= ~reinterpret_cast<uint64_t*>(ballot)[0];
            gtMask31 &= ~reinterpret_cast<uint64_t*>(ballot)[1];
        }
    }
}

//Asm 20% faster
template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void MultiSplit128Asm(
    uint32_t& eqMask00, uint64_t& eqMask10, uint64_t& eqMask20, uint32_t& eqMask21,
    uint64_t& eqMask30, uint64_t& eqMask31, uint64_t& gtMask00, uint64_t& gtMask01,
    uint64_t& gtMask10, uint64_t& gtMask11, uint64_t& gtMask20, uint64_t& gtMask21,
    uint64_t& gtMask30, uint64_t& gtMask31,
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
            "    .reg .b64 inv;\n"
            "    and.b32 bal0, %14, %18;\n"
            "    setp.ne.u32 p0, bal0, 0;\n"
            "    and.b32 bal1, %15, %18;\n"
            "    setp.ne.u32 p1, bal1, 0;\n"
            "    and.b32 bal2, %16, %18;\n"
            "    setp.ne.u32 p2, bal2, 0;\n"
            "    and.b32 %18, %17, %18;\n"
            "    setp.ne.u32 p3, %18, 0;\n"
            "    vote.ballot.sync.b32 bal0, p0, 0xffffffff;\n"
            "    vote.ballot.sync.b32 bal1, p1, 0xffffffff;\n"
            "    vote.ballot.sync.b32 bal2, p2, 0xffffffff;\n"
            "    vote.ballot.sync.b32 %18, p3, 0xffffffff;\n"
            "    mov.b64 inv, {bal0, bal1};\n"
            "    @p0 and.b32 %0, %0, bal0;\n"
            "    @p1 and.b64 %1, %1, inv;\n"
            "    @p2 and.b64 %2, %2, inv;\n"
            "    @p2 and.b32 %3, %3, bal2;\n"
            "    @p3 and.b64 %4, %4, inv;\n"
            "    not.b64 inv, inv;\n"
            "    @p0 or.b64 %6, %6, inv;\n"
            "    @p1 or.b64 %8, %8, inv;\n"
            "    @p2 or.b64 %10, %10, inv;\n"
            "    @p3 or.b64 %12, %12, inv;\n"
            "    @!p0 and.b64 %6, %6, inv;\n"
            "    @!p1 and.b64 %8, %8, inv;\n"
            "    @!p2 and.b64 %10, %10, inv;\n"
            "    @!p3 and.b64 %12, %12, inv;\n"
            "    cvt.u32.u64 bal0, inv;\n"
            "    @!p1 and.b64 %1, %1, inv;\n"
            "    @!p2 and.b64 %2, %2, inv;\n"
            "    @!p3 and.b64 %4, %4, inv;\n"
            "    mov.b64 inv, {bal2, %18};\n"
            "    @p3 and.b64 %5, %5, inv;\n"
            "    not.b64 inv, inv;\n"
            "    @p0 or.b64 %7, %7, inv;\n"
            "    @p1 or.b64 %9, %9, inv;\n"
            "    @p2 or.b64 %11, %11, inv;\n"
            "    @p3 or.b64 %13, %13, inv;\n"
            "    @!p0 and.b64 %7, %7, inv;\n"
            "    @!p1 and.b64 %9, %9, inv;\n"
            "    @!p2 and.b64 %11, %11, inv;\n"
            "    @!p3 and.b64 %13, %13, inv;\n"
            "    cvt.u32.u64 bal2, inv;\n"
            "    @!p0 and.b32 %0, %0, bal0;\n"
            "    @!p2 and.b32 %3, %3, bal2;\n"
            "    @!p3 and.b64 %5, %5, inv;\n"
            "}\n" : "+r"(eqMask00), "+l"(eqMask10), "+l"(eqMask20), "+r"(eqMask21),
            "+l"(eqMask30), "+l"(eqMask31), "+l"(gtMask00), "+l"(gtMask01),
            "+l"(gtMask10), "+l"(gtMask11), "+l"(gtMask20), "+l"(gtMask21),
            "+l"(gtMask30), "+l"(gtMask31) :
            "r"(key0), "r"(key1), "r"(key2), "r"(key3),
            "r"(current_bit));
    }

    //less verbose asm, faster than no asm, slower than full asm
    /*uint32_t ballot[4];
    uint16_t setBit[4];
    asm("{\n"
        "    .reg .pred p0;\n"
        "    .reg .pred p1;\n"
        "    .reg .pred p2;\n"
        "    .reg .pred p3;\n"
        "    and.b32 %4, %8, %12;\n"
        "    setp.ne.u32 p0, %4, 0;\n"
        "    vote.ballot.sync.b32 %4, p0, 0xffffffff;\n"
        "    and.b32 %5, %9, %12;\n"
        "    setp.ne.u32 p1, %5, 0;\n"
        "    vote.ballot.sync.b32 %5, p1, 0xffffffff;\n"
        "    and.b32 %6, %10, %12;\n"
        "    setp.ne.u32 p2, %6, 0;\n"
        "    vote.ballot.sync.b32 %6, p2, 0xffffffff;\n"
        "    and.b32 %7, %11, %12;\n"
        "    setp.ne.u32 p3, %7, 0;\n"
        "    vote.ballot.sync.b32 %7, p3, 0xffffffff;\n"
        "    selp.u16 %0, 1, 0, p0;\n"
        "    selp.u16 %1, 1, 0, p1;\n"
        "    selp.u16 %2, 1, 0, p2;\n"
        "    selp.u16 %3, 1, 0, p3;\n"
        "}\n" : "=h"(setBit[0]), "=h"(setBit[1]), "=h"(setBit[2]), "=h"(setBit[3]),
                "=r"(ballot[0]), "=r"(ballot[1]), "=r"(ballot[2]), "=r"(ballot[3])  :
        "r"(keys[k]), "r"(keys[k + 1]), "r"(keys[k + 2]), "r"(keys[k + 3]),
        "r"(current_bit));

        . . .
    */
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
__device__ __forceinline__ void cs128(
    uint32_t& key0, uint32_t& key1, uint32_t& key2, uint32_t& key3,
    uint2* s_pairs,
    const uint32_t totalLocalLength,
    const uint32_t runStart)
{
    if (totalLocalLength - runStart > 64)
    {
        uint32_t eqMask00 = ~0U;
        uint64_t eqMask10 = ~0ULL;
        uint64_t eqMask20 = ~0ULL;
        uint32_t eqMask21 = ~0U;
        uint64_t eqMask30 = ~0ULL;
        uint64_t eqMask31 = ~0ULL;

        uint64_t gtMask00 = 0;
        uint64_t gtMask01 = 0;
        uint64_t gtMask10 = 0;
        uint64_t gtMask11 = 0;
        uint64_t gtMask20 = 0;
        uint64_t gtMask21 = 0;
        uint64_t gtMask30 = 0;
        uint64_t gtMask31 = 0;

        MultiSplit128Asm<BITS_TO_SORT>(
            eqMask00, eqMask10, eqMask20, eqMask21,
            eqMask30, eqMask31,
            gtMask00, gtMask01, gtMask10, gtMask11,
            gtMask20, gtMask21, gtMask30, gtMask31,
            key0, key1, key2, key3);

        {
            uint32_t t = __popcll(gtMask00) + __popcll(gtMask01);
            t += __popc(eqMask00 & getLaneMaskLt());
            s_pairs[t] = { key0, getLaneId() + runStart };
        }

        {
            uint32_t t = __popcll(gtMask10) + __popcll(gtMask11);
            t += __popcll(eqMask10 & ((uint64_t)getLaneMaskLt() << 32ULL | 0xffffffff));
            s_pairs[t] = { key1, getLaneId() + runStart + 32 };
        }

        if (getLaneId() + runStart + 64 < totalLocalLength)
        {
            uint32_t t = __popcll(gtMask20) + __popcll(gtMask21);
            t += __popcll(eqMask20) + __popc(eqMask21 & getLaneMaskLt());
            s_pairs[t] = { key2, getLaneId() + runStart + 64 };
        }
        else
        {
            s_pairs[getLaneId() + 64].x = 0xffffffff;
        }

        if (getLaneId() + runStart + 96 < totalLocalLength)
        {
            uint32_t t = __popcll(gtMask30) + __popcll(gtMask31);
            t += __popcll(eqMask30) + __popcll(eqMask31 & ((uint64_t)getLaneMaskLt() << 32ULL | 0xffffffff));
            s_pairs[t] = { key3, getLaneId() + runStart + 96 };
        }
        else
        {
            s_pairs[getLaneId() + 96].x = 0xffffffff;
        }
    }
    else
    {
        if (totalLocalLength - runStart > 32)
        {
            cs64<BITS_TO_SORT>(
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
            cs32<BITS_TO_SORT>(
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

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void cs128Ge(
    uint32_t& key0, uint32_t& key1, uint32_t& key2, uint32_t& key3,
    uint2* s_pairs,
    const uint32_t totalLocalLength,
    const uint32_t runStart)
{
    if (totalLocalLength - runStart > 64)
    {
        const uint64_t upperMask = (uint64_t)getLaneMaskLt() << 32ULL | 0xffffffff;
        uint64_t geMask00 = getLaneMaskLt();
        uint64_t geMask01 = 0;
        uint64_t geMask10 = upperMask;
        uint64_t geMask11 = 0;
        uint64_t geMask20 = ~0ULL;
        uint64_t geMask21 = getLaneMaskLt();
        uint64_t geMask30 = ~0ULL;
        uint64_t geMask31 = upperMask;

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

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void MultiSplit256Ge(
    uint64_t& geMask00, uint64_t& geMask10, uint64_t& geMask20, uint64_t& geMask30,
    uint64_t& geMask40, uint64_t& geMask50, uint64_t& geMask60, uint64_t& geMask70,
    uint64_t& geMask01, uint64_t& geMask11, uint64_t& geMask21, uint64_t& geMask31,
    uint64_t& geMask41, uint64_t& geMask51, uint64_t& geMask61, uint64_t& geMask71,
    const uint32_t key0, const uint32_t key1, const uint32_t key2, const uint32_t key3,
    const uint32_t key4, const uint32_t key5, const uint32_t key6, const uint32_t key7)
{
    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
    {
        uint32_t current_bit = 1 << bit;
        uint32_t bal0;
        uint32_t bal1;
        uint32_t bal2;
        bool p0;
        bool p1;
        bool p2;
        bool p3;
        bool p4;
        bool p5;
        bool p6;
        bool p7;

        bal0 = key4 & current_bit;
        p4 = bal0 == 0;
        bal0 = key5 & current_bit;
        p5 = bal0 == 0;
        bal0 = key6 & current_bit;
        p6 = bal0 == 0;
        bal0 = key7 & current_bit;
        p7 = bal0 == 0;

        bal0 = key0 & current_bit;
        p0 = bal0 == 0;
        bal1 = key1 & current_bit;
        p1 = bal1 == 0;
        bal2 = key2 & current_bit;
        p2 = bal2 == 0;
        current_bit &= key3;
        p3 = current_bit == 0;

        bal0 = __ballot_sync(0xffffffff, p0);
        bal1 = __ballot_sync(0xffffffff, p1);
        bal2 = __ballot_sync(0xffffffff, p2);
        current_bit = __ballot_sync(0xffffffff, p3);

        uint64_t t;
        asm("mov.b64 %0, {%1, %2};" : "=l"(t) : "r"(bal0), "r"(bal1));
        if (p0) geMask00 &= t; else geMask00 |= t;
        if (p1) geMask10 &= t; else geMask10 |= t;
        if (p2) geMask20 &= t; else geMask20 |= t;
        if (p3) geMask30 &= t; else geMask30 |= t;
        if (p4) geMask40 &= t; else geMask40 |= t;
        if (p5) geMask50 &= t; else geMask50 |= t;
        if (p6) geMask60 &= t; else geMask60 |= t;
        if (p7) geMask70 &= t; else geMask70 |= t;

        asm("mov.b64 %0, {%1, %2};" : "=l"(t) : "r"(bal2), "r"(current_bit));
        if (p0) geMask01 &= t; else geMask01 |= t;
        if (p1) geMask11 &= t; else geMask11 |= t;
        if (p2) geMask21 &= t; else geMask21 |= t;
        if (p3) geMask31 &= t; else geMask31 |= t;
        if (p4) geMask41 &= t; else geMask41 |= t;
        if (p5) geMask51 &= t; else geMask51 |= t;
        if (p6) geMask61 &= t; else geMask61 |= t;
        if (p7) geMask71 &= t; else geMask71 |= t;
    }
}

template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void cs256Ge(
    uint32_t& key0, uint32_t& key1, uint32_t& key2, uint32_t& key3,
    uint32_t& key4, uint32_t& key5, uint32_t& key6, uint32_t& key7,
    uint2* s_pairs,
    const uint32_t totalLocalLength,
    const uint32_t runStart)
{
    if (totalLocalLength - runStart > 128)
    {
        uint32_t indexes[8];
        const uint64_t upperMask = (uint64_t)getLaneMaskLt() << 32ULL | 0xffffffff;
        
        uint64_t geMask00 = getLaneMaskLt();
        uint64_t geMask10 = upperMask;
        uint64_t geMask20 = ~0ULL;
        uint64_t geMask30 = ~0ULL;
        uint64_t geMask40 = ~0ULL;
        uint64_t geMask50 = ~0ULL;
        uint64_t geMask60 = ~0ULL;
        uint64_t geMask70 = ~0ULL;

        uint64_t geMask01 = 0;
        uint64_t geMask11 = 0;
        uint64_t geMask21 = getLaneMaskLt();
        uint64_t geMask31 = upperMask;
        uint64_t geMask41 = ~0ULL;
        uint64_t geMask51 = ~0ULL;
        uint64_t geMask61 = ~0ULL;
        uint64_t geMask71 = ~0ULL;

        MultiSplit256Ge<BITS_TO_SORT>(
            geMask00, geMask10, geMask20, geMask30,
            geMask40, geMask50, geMask60, geMask70,
            geMask01, geMask11, geMask21, geMask31,
            geMask41, geMask51, geMask61, geMask71,
            key0, key1, key2, key3,
            key4, key5, key6, key7);

        indexes[0] = __popcll(geMask00) + __popcll(geMask01);
        indexes[1] = __popcll(geMask10) + __popcll(geMask11);
        indexes[2] = __popcll(geMask20) + __popcll(geMask21);
        indexes[3] = __popcll(geMask30) + __popcll(geMask31);
        indexes[4] = __popcll(geMask40) + __popcll(geMask41);
        indexes[5] = __popcll(geMask50) + __popcll(geMask51);
        indexes[6] = __popcll(geMask60) + __popcll(geMask61);
        indexes[7] = __popcll(geMask70) + __popcll(geMask71);

        geMask00 = 0;
        geMask10 = 0;
        geMask20 = 0;
        geMask30 = 0;
        geMask40 = getLaneMaskLt();
        geMask50 = upperMask;
        geMask60 = ~0ULL;
        geMask70 = ~0ULL;

        geMask01 = 0;
        geMask11 = 0;
        geMask21 = 0;
        geMask31 = 0;
        geMask41 = 0;
        geMask51 = 0;
        geMask61 = getLaneMaskLt();
        geMask71 = upperMask;

        MultiSplit256Ge<BITS_TO_SORT>(
            geMask40, geMask50, geMask60, geMask70,     //Note the shuffled order
            geMask00, geMask10, geMask20, geMask30,
            geMask41, geMask51, geMask61, geMask71,
            geMask01, geMask11, geMask21, geMask31,
            key4, key5, key6, key7,
            key0, key1, key2, key3);

        indexes[0] += __popcll(geMask00) + __popcll(geMask01);
        indexes[1] += __popcll(geMask10) + __popcll(geMask11);
        indexes[2] += __popcll(geMask20) + __popcll(geMask21);
        indexes[3] += __popcll(geMask30) + __popcll(geMask31);
        indexes[4] += __popcll(geMask40) + __popcll(geMask41);
        indexes[5] += __popcll(geMask50) + __popcll(geMask51);
        indexes[6] += __popcll(geMask60) + __popcll(geMask61);
        indexes[7] += __popcll(geMask70) + __popcll(geMask71);

        s_pairs[indexes[0]] = { key0, getLaneId() + runStart };
        s_pairs[indexes[1]] = { key1, getLaneId() + runStart + 32 };
        s_pairs[indexes[2]] = { key2, getLaneId() + runStart + 64 };
        s_pairs[indexes[3]] = { key3, getLaneId() + runStart + 96 };
        s_pairs[indexes[4]] = { key4, getLaneId() + runStart + 128 };
        s_pairs[indexes[5]] = { key5, getLaneId() + runStart + 160 };
        s_pairs[indexes[6]] = { key6, getLaneId() + runStart + 192 };
        s_pairs[indexes[7]] = { key7, getLaneId() + runStart + 224 };
    }
    else
    {
        if (totalLocalLength - runStart > 64)
        {
            cs128Ge<BITS_TO_SORT>(
                key0,
                key1,
                key2,
                key3,
                s_pairs,
                totalLocalLength,
                runStart);
            s_pairs[getLaneId() + 128].x = 0xffffffff;
            s_pairs[getLaneId() + 160].x = 0xffffffff;
            s_pairs[getLaneId() + 192].x = 0xffffffff;
            s_pairs[getLaneId() + 224].x = 0xffffffff;
        }
        else if (totalLocalLength - runStart > 32)
        {
            cs64Ge<BITS_TO_SORT>(
                key0,
                key1,
                s_pairs,
                totalLocalLength,
                runStart);
            s_pairs[getLaneId() + 64].x = 0xffffffff;
            s_pairs[getLaneId() + 96].x = 0xffffffff;
            s_pairs[getLaneId() + 128].x = 0xffffffff;
            s_pairs[getLaneId() + 160].x = 0xffffffff;
            s_pairs[getLaneId() + 192].x = 0xffffffff;
            s_pairs[getLaneId() + 224].x = 0xffffffff;
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
            s_pairs[getLaneId() + 128].x = 0xffffffff;
            s_pairs[getLaneId() + 160].x = 0xffffffff;
            s_pairs[getLaneId() + 192].x = 0xffffffff;
            s_pairs[getLaneId() + 224].x = 0xffffffff;
        }
    }
}

//redundant work is way too slow
template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
__device__ __forceinline__ void CuteSort256(
    uint32_t* keys,
    uint2* s_pairs,
    const uint32_t totalLocalLength,
    const uint32_t warpOffset)
{
    #pragma unroll
    for (uint32_t k = 0; k < KEYS_PER_THREAD; k += 8)
    {
        const uint32_t runStart = k * LANE_COUNT + warpOffset;
        if (runStart < totalLocalLength)
        {
            cs256Ge<BITS_TO_SORT>(
                keys[k], keys[k + 1], keys[k + 2], keys[k + 3],
                keys[k + 4], keys[k + 5], keys[k + 6], keys[k + 7],
                &s_pairs[k >> 3 << 8],
                totalLocalLength,
                runStart);
        }
        else
        {
            s_pairs[getLaneId() + k * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 1) * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 2) * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 3) * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 4) * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 5) * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 6) * LANE_COUNT].x = 0xffffffff;
            s_pairs[getLaneId() + (k + 7) * LANE_COUNT].x = 0xffffffff;
        }
    }
}

__device__ __forceinline__ void LoadBins(
    const uint32_t* segments,
    uint32_t* s_warpBins,
    const uint32_t binCount,
    const uint32_t binOffset,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    if (getLaneId() <= binCount)
    {
        s_warpBins[getLaneId()] = getLaneId() + binOffset >= totalSegCount ?
            totalSegLength : segments[getLaneId() + binOffset];
    }
    __syncwarp(0xffffffff);
}

__device__ __forceinline__ BinInfo32 GetBinInfo32(
    const uint32_t* s_warpBins,
    const uint32_t binCount)
{
    const uint2 interval = BBUtils::BinarySearch(s_warpBins, (int32_t)binCount, getLaneId());
    const uint32_t binMask = (((1ULL << interval.y) - 1) >> interval.x << interval.x);
    return BinInfo32{ binMask, interval.x };
}

__device__ __forceinline__ BinInfo64 GetBinInfo64(
    const uint32_t* s_warpBins,
    const uint32_t binCount,
    const uint32_t targetIndex)
{
    const uint2 interval = BBUtils::BinarySearch(s_warpBins, (int32_t)binCount, targetIndex);
    const uint64_t binMask = (((1ULL << interval.y) - 1) >> interval.x << interval.x);
    return BinInfo64{ binMask, interval.x };
}

__device__ __forceinline__ void SingleBinFallback(
    uint32_t& key,
    uint32_t& index,
    const uint32_t totalLocalLength)
{
    index = getLaneId();
    BBUtils::RegSortFallback(key, index, totalLocalLength);
}

// Only a single warp participates in sorting a run of keys
// However, multiple independent warps can be launched in a single block
template<
    uint32_t WARP_KEYS,
    uint32_t BLOCK_KEYS,
    uint32_t WARPS,
    uint32_t BITS_TO_SORT,
    class K>
__device__ __forceinline__ void SplitSortBins32(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    const uint32_t* minBinSegCounts,
    uint32_t* sort,
    K* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength,
    const uint32_t segCountInBin)
{
    if (blockIdx.x * WARPS + WARP_INDEX >= segCountInBin)
        return;

    __shared__ uint32_t s_bins[LANE_COUNT * WARPS];
    uint32_t* s_warpBins = &s_bins[WARP_INDEX * LANE_COUNT];

    const uint32_t binOffset = binOffsets[blockIdx.x * WARPS + WARP_INDEX];
    const uint32_t binCount = minBinSegCounts[blockIdx.x * WARPS + WARP_INDEX];
    
    //If the bin count is 32, then all bins must be of size 1, so we short circuit
    if (binCount == 32)
        return;

    LoadBins(segments, s_warpBins, binCount, binOffset, totalSegCount, totalSegLength);
    const uint32_t segmentStart = s_warpBins[0];
    const uint32_t totalLocalLength = s_warpBins[binCount] - segmentStart;
    uint32_t key = getLaneId() < totalLocalLength ? sort[getLaneId() + segmentStart] : 0xffffffff;
    
    //If the binCount is 1, and the length of the segment
    //is short, skip cute sort and use a regSort style fallback
    uint32_t index;
    K val;
    if (binCount == 1 && totalLocalLength <= 16)
    {
        SingleBinFallback(key, index, totalLocalLength);
        if (getLaneId() < totalLocalLength)
            val = payloads[index + segmentStart];
        __syncwarp(0xffffffff);
        if (getLaneId() < totalLocalLength)
        {
            sort[getLaneId() + segmentStart] = key;
            payloads[getLaneId() + segmentStart] = val;
        }
    }
    else
    {
        CuteSort32BinGe<BITS_TO_SORT>(key, index, GetBinInfo32(s_warpBins, binCount), totalLocalLength);
        if (getLaneId() < totalLocalLength)
            val = payloads[getLaneId() + segmentStart];
        __syncwarp(0xffffffff);
        if (getLaneId() < totalLocalLength)
        {
            sort[index + segmentStart] = key;
            payloads[index + segmentStart] = val;
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
        bool pred = startB >= (mergeLength << 1) || (startA < mergeLength && t0.x <= t1.x);
        
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
        const uint32_t startA = BBUtils::find_kth3(
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
    class K>
__device__ __forceinline__ void SplitSortWarp(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    K* payloads,
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

    uint32_t keys[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + segmentStart, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        keys[k] = i < segmentEnd ? sort[i] : 0xffffffff;
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
    K vals[KEYS_PER_THREAD];
    if (WARP_LOG_END == WARP_LOG_START)
    {
        #pragma unroll
        for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                sort[i + segmentStart] = s_warpPairs[i].x;
        }

        #pragma unroll
        for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                vals[k]= payloads[s_warpPairs[i].y + segmentStart];
        }
        __syncwarp(0xffffffff);

        #pragma unroll
        for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                payloads[i + segmentStart] = vals[k];
        }
    }

    //Else, scatter the post merge results from registers, for most
    //warp sized partition workloads, prescattering to shared memory is a slowdown
    if (WARP_LOG_END > WARP_LOG_START)
    {
        #pragma unroll
        for (uint32_t i = getLaneId() * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
        {
            if (i < totalLocalLength)
                sort[i + segmentStart] = pairs[k].x;
        }

        #pragma unroll
        for (uint32_t i = getLaneId() * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[pairs[k].y + segmentStart];
        }
        __syncwarp(0xffffffff);

        #pragma unroll
        for (uint32_t i = getLaneId() * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
        {
            if (i < totalLocalLength)
                payloads[i + segmentStart] = vals[k];
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
    class K>
__device__ __forceinline__ void SplitSortBlock(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    K* payloads,
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

    uint32_t keys[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * WARP_KEYS, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        keys[k] = i < totalLocalLength ? sort[i + segmentStart] : 0xffffffff;
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

    K vals[KEYS_PER_THREAD];
    if (SHOULD_PRE_SCATTER)
    {
        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if (i < totalLocalLength)
                sort[i + segmentStart] = s_blockPairs[i].x;
        }

        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[s_blockPairs[i].y + segmentStart];
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if (i < totalLocalLength)
                payloads[i + segmentStart] = vals[k];
        }
    }

    if (!SHOULD_PRE_SCATTER)
    {
        #pragma unroll
        for (uint32_t i = threadIdx.x * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
        {
            if (i < totalLocalLength)
                sort[i + segmentStart] = pairs[k].x;
        }

        #pragma unroll
        for (uint32_t i = threadIdx.x * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[pairs[k].y + segmentStart];
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t i = threadIdx.x * KEYS_PER_THREAD, k = 0; k < KEYS_PER_THREAD; ++i, ++k)
        {
            if (i < totalLocalLength)
                payloads[i + segmentStart] = vals[k];
        }
    }    
}

template<uint32_t BITS_TO_RANK>
__device__ __forceinline__ void MultiSplitRadix(
    uint32_t& eqMask,
    const uint32_t key,
    const uint32_t radixShift)
{
    eqMask = 0xffffffff;
    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_RANK; ++bit)
    {
        const bool isBitSet = ExtractDigit(key, bit + radixShift);
        unsigned ballot = __ballot_sync(0xffffffff, isBitSet);
        eqMask &= isBitSet ? ballot : ~ballot;
    }
}

template<uint32_t BITS_TO_RANK>
__device__ __forceinline__ void MultiSplitRadixAsm(
    uint32_t& eqMask,
    const uint32_t key,
    const uint32_t radixShift)
{
    #pragma unroll
    for (uint32_t bit = 0; bit < BITS_TO_RANK; ++bit)
    {
        uint32_t mask;
        uint32_t current_bit = 1 << bit + radixShift;
        asm("{\n"
            "    .reg .pred p;\n"
            "    and.b32 %0, %1, %2;"
            "    setp.ne.u32 p, %0, 0;\n"
            "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
            "    @!p not.b32 %0, %0;\n"
            "}\n" : "=r"(mask) : "r"(key), "r"(current_bit));
        eqMask = (bit == 0) ? mask : eqMask & mask;
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

template<
    uint32_t WARPS,
    uint32_t KEYS_PER_THREAD,
    uint32_t KEYS_PER_WARP,
    uint32_t PART_SIZE,
    uint32_t BITS_TO_SORT,
    uint32_t RADIX,
    uint32_t RADIX_MASK,
    uint32_t RADIX_LOG,
    class K>
__device__ __forceinline__ void SplitSortRadix(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    K* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    __shared__ uint32_t s_mem[RADIX * WARPS + PART_SIZE]; //Conveniently these are equal for 256/2048 and 512/4096
    uint32_t* s_hist = s_mem;
    uint32_t* s_indexes = &s_mem[RADIX * WARPS];

    const uint32_t binOffset = binOffsets[blockIdx.x];
    const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
    const uint32_t segmentStart = segments[binOffset];
    const uint32_t totalLocalLength = segmentEnd - segmentStart;

    uint32_t* s_warpHist = &s_hist[WARP_INDEX * RADIX];
    ClearWarpHist<RADIX>(s_warpHist);

    uint32_t keys[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        keys[k] = i < totalLocalLength ? sort[i + segmentStart] : 0xffffffff;
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

        if (threadIdx.x < (RADIX >> LANE_LOG))
            s_hist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_hist[threadIdx.x << LANE_LOG]);
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
        if(i < totalLocalLength)
            sort[i + segmentStart] = s_hist[i];
    }

    //If possible, scatter the payloads into shared memory prior to device
    K vals[KEYS_PER_THREAD];
    if (sizeof(K) * PART_SIZE <= (RADIX * WARPS + PART_SIZE) * sizeof(uint32_t))
    {
        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[i + segmentStart];
        }

        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
            indexes[k] = s_indexes[indexes[k]];
        __syncthreads();

        K* s_payloadsOut = reinterpret_cast<K*>(s_mem);
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
                payloads[i + segmentStart] = s_payloadsOut[i];
        }
    }

    if (sizeof(K) * PART_SIZE > (RADIX * WARPS + PART_SIZE) * sizeof(uint32_t))
    {
        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[i + segmentStart];
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                payloads[s_indexes[indexes[k]] + segmentStart] = vals[k];
        }
    }
}

//Get the totalLocalLength of a segment and advance
//the device pointers to the correction location
template<class K>
__device__ __forceinline__ void GetSegmentInfoRadixFine(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t*& sort,
    K*& payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength,
    uint32_t& totalLocalLength)
{
    const uint32_t binOffset = binOffsets[blockIdx.x];
    const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
    const uint32_t segmentStart = segments[binOffset];
    totalLocalLength = segmentEnd - segmentStart;
    sort += segmentStart;
    payloads += segmentStart;
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
    class K>
__device__ __forceinline__ void SplitSortRadixFine(
    uint32_t* s_hist,
    uint32_t* s_indexes,
    uint32_t* sort,
    K* payloads,
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

        if (threadIdx.x < (RADIX >> LANE_LOG))
            s_hist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_hist[threadIdx.x << LANE_LOG]);
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

    //If possible, scatter the payloads into shared memory prior to device
    K vals[KEYS_PER_THREAD];
    if (sizeof(K) * PART_SIZE <= (RADIX * WARPS + PART_SIZE) * sizeof(uint32_t))
    {
        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[i];
        }

        for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
            indexes[k] = s_indexes[indexes[k]];
        __syncthreads();

        K* s_payloadsOut = reinterpret_cast<K*>(s_hist);
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
                payloads[i] = s_payloadsOut[i];
        }
    }

    if (sizeof(K) * PART_SIZE > (RADIX * WARPS + PART_SIZE) * sizeof(uint32_t))
    {
        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                vals[k] = payloads[i];
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
            k < KEYS_PER_THREAD;
            i += LANE_COUNT, ++k)
        {
            if (i < totalLocalLength)
                payloads[s_indexes[indexes[k]]] = vals[k];
        }
    }
}

//Raw counting sort for unique keys < radix. Turns out to be slower than the tradtional radix
template<
    uint32_t WARPS,
    uint32_t KEYS_PER_THREAD,
    uint32_t RADIX,
    uint32_t WARP_PARTS,
    uint32_t WARP_PART_SIZE,
    bool SHOULD_PRE_SCATTER,
    class K>
__device__ __forceinline__ void  CountSort(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    K* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    __shared__ uint16_t s_mem[RADIX];

    const uint32_t binOffset = binOffsets[blockIdx.x];
    const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
    const uint32_t segmentStart = segments[binOffset];
    const uint32_t totalLocalLength = segmentEnd - segmentStart;

    for (uint32_t i = threadIdx.x; i < RADIX / 8; i += blockDim.x)
        reinterpret_cast<uint64_t*>(s_mem)[i] = 0;

    uint32_t keys[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
    {
        if(i < totalLocalLength)
            keys[k] = sort[i + segmentStart];
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
    {
        if (i < totalLocalLength)
            s_mem[keys[k]]++;
    }
    __syncthreads();
    
    uint32_t reduction = 0;
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * WARP_PART_SIZE, k = 0;
        k < WARP_PARTS;
        i += LANE_COUNT, ++k)
    {
        const uint32_t t = InclusiveWarpScan(s_mem[i]) + reduction;
        s_mem[i] = t;
        reduction = __shfl_sync(0xffffffff, t, LANE_MASK);
    }
    __syncthreads();

    if (threadIdx.x < WARPS)
        s_mem[(threadIdx.x + 1) * WARP_PART_SIZE - 1] = ActiveInclusiveWarpScan(s_mem[(threadIdx.x + 1) * WARP_PART_SIZE - 1]);
    __syncthreads();

    uint32_t prev = threadIdx.x >= LANE_COUNT ? s_mem[WARP_INDEX * WARP_PART_SIZE - 1] : 0;
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * WARP_PART_SIZE, k = 0;
        k < WARP_PARTS;
        i += LANE_COUNT, ++k)
    {
        s_mem[i] += prev;
    }
    __syncthreads();

    uint32_t offsets[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
    {
        if (i < totalLocalLength)
            offsets[k] = keys[k] ? s_mem[keys[k] - 1] : 0;
    }
    __syncthreads();

    K vals[KEYS_PER_THREAD];
    if (SHOULD_PRE_SCATTER)
    {

    }

    if (!SHOULD_PRE_SCATTER)
    {
        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if (i < totalLocalLength)
            {
                sort[offsets[k] + segmentStart] = keys[k];
                vals[k] = payloads[i + segmentStart];
            }
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
        {
            if (i < totalLocalLength)
                payloads[offsets[k] + segmentStart] = vals[k];
        }
    }
}

namespace SplitSortVariants
{
    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv32_cute32_bin(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* minBinSegCounts,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBins32<32, 32 * WARPS, WARPS, BITS_TO_SORT, K>(
            segments,
            binOffsets,
            minBinSegCounts,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv64_cute32_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<2, 64, 64 * WARPS, WARPS, 5, 6, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv64_cute64_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<2, 64, 64 * WARPS, WARPS, 6, 6, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv128_cute32_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<4, 128, 128 * WARPS, WARPS, 5, 7, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv128_cute64_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<4, 128, 128 * WARPS, WARPS, 6, 7, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv128_cute128_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<4, 128, 128 * WARPS, WARPS, 7, 7, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort128<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv128_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 128, 2, 5, 6, 7, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv128_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 128, 2, 6, 6, 7, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv256_cute32_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 5, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t32_kv256_cute64_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 6, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort64<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
        __global__ void t32_kv256_cute128_wMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 7, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort128<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
        __global__ void t32_kv256_cute256_wMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 8, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort256<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv256_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 256, 2, 5, 7, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv256_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 256, 2, 6, 7, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv256_cute128_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 256, 2, 7, 7, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort128<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv256_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 256, 4, 5, 6, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv256_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 256, 4, 6, 6, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t256_kv256_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<1, 32, 256, 8, 5, 5, 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 1>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv512_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 512, 2, 5, 8, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv512_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 512, 4, 5, 7, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t256_kv512_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 512, 8, 5, 6, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv512_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 512, 2, 6, 8, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv512_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 512, 4, 6, 7, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t256_kv512_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 512, 8, 6, 6, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t64_kv512_cute128_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 512, 2, 7, 8, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort128<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv512_cute128_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 512, 4, 7, 7, 9, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort128<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv1024_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 1024, 4, 6, 8, 10, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t256_kv1024_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 1024, 8, 6, 7, 10, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t128_kv1024_cute128_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 1024, 4, 7, 8, 10, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort128<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t256_kv1024_cute128_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 1024, 8, 7, 7, 10, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort128<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
        __global__ void t128_kv1024_cute256_bMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 1024, 4, 8, 8, 10, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort256<BITS_TO_SORT, 8>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t512_kv2048_cute64_bMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 2048, 16, 6, 7, 11, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t512_kv2048_cute128_bMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortBlock<4, 128, 2048, 16, 7, 7, 11, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort128<BITS_TO_SORT, 4>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
    __global__ void t1024_kv2048_cute64_bMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortBlock<2, 64, 2048, 16, 6, 6, 11, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 2>);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT,
        class K>
        __global__ void t512_kv4096_cute256_bMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            K* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength,
            const uint32_t segCountInBin)
    {
        SplitSortBlock<8, 256, 4096, 16, 8, 8, 12, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort256<BITS_TO_SORT, 8>);
    }

    //RADIX SORTS
    #define ROUND_UP_BITS_TO_SORT   ((BITS_TO_SORT >> 3) + ((BITS_TO_SORT & 7) ? 1 : 0) << 3)

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t64_kv128_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<2, 2, 64, 128, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t64_kv256_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<2, 4, 128, 256, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t64_kv512_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<2, 8, 256, 512, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t128_kv512_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<4, 4, 128, 512, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t256_kv512_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<8, 2, 64, 512, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t128_kv1024_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<4, 8, 256, 1024, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t256_kv1024_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<8, 4, 128, 1024, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t256_kv2048_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<8, 8, 256, 2048, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t512_kv2048_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<16, 4, 128, 2048, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void  t512_kv4096_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin) 
    {
        SplitSortRadix<16, 8, 256, 4096, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t1024_kv4096_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortRadix<32, 4, 128, 4096, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    //Fine granularity radix sorts
    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t128_kv1024_radixFine(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        __shared__ uint32_t s_mem[256 * 4 + 1024];  //RADIX * 4 + PART_SIZE
        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        payloads += segmentStart;

        if (totalLocalLength <= 640)
        {
            SplitSortRadixFine<4, 5, 160, 640, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[1024], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }
        
        if (totalLocalLength > 640 && totalLocalLength <= 768)
        {
            SplitSortRadixFine<4, 6, 192, 768, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[1024], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 768 && totalLocalLength <= 896)
        {
            SplitSortRadixFine<4, 7, 224, 896, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[1024], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 896)
        {
            SplitSortRadixFine<4, 8, 256, 1024, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[1024], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void t256_kv2048_radixFine(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        __shared__ uint32_t s_mem[256 * 8 + 2048];  //RADIX * WARPS + PART_SIZE
        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        payloads += segmentStart;

        if (totalLocalLength <= 1280)
        {
            SplitSortRadixFine<8, 5, 160, 1280, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 1280 && totalLocalLength <= 1536)
        {
            SplitSortRadixFine<8, 6, 192, 1536, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 1536 && totalLocalLength <= 1792)
        {
            SplitSortRadixFine<8, 7, 224, 1792, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 1792)
        {
            SplitSortRadixFine<8, 8, 256, 2048, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }
    }

    template<uint32_t BITS_TO_SORT, class K>
    __global__ void  t512_kv4096_radixFine(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        __shared__ uint32_t s_mem[256 * 16 + 4096];  //RADIX * WARPS + PART_SIZE
        const uint32_t binOffset = binOffsets[blockIdx.x];
        const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
        const uint32_t segmentStart = segments[binOffset];
        const uint32_t totalLocalLength = segmentEnd - segmentStart;
        sort += segmentStart;
        payloads += segmentStart;

        if (totalLocalLength <= 2560)
        {
            SplitSortRadixFine<16, 5, 160, 2560, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 2560 && totalLocalLength <= 3072)
        {
            SplitSortRadixFine<16, 6, 192, 3072, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 3072 && totalLocalLength <= 3584)
        {
            SplitSortRadixFine<16, 7, 224, 3584, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }

        if (totalLocalLength > 3584)
        {
            SplitSortRadixFine<16, 8, 256, 4096, ROUND_UP_BITS_TO_SORT, 256, 255, 8, K>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                payloads,
                totalLocalLength);
        }
    }

    //Hack sorts, unique keys && keys < 8192 only
    template<class K>
    __global__ void t256_kv512_count(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        K* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        CountSort<8, 2, 4096, 4096 / 8 / LANE_COUNT, 4096 / 8, false, K>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }
}