/******************************************************************************
*  GPUSorting
 * SplitSort
 * Experimental SegSort that does not use cooperative groups
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 5/16/2024
 * https://github.com/b0nes164/GPUSorting
 *
 * Using "CuteSort" technique by
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
struct BinInfo32
{
    uint32_t binMask;
    uint32_t binOffset;
};

struct BinInfo64
{
    uint32_t binMask0;
    uint64_t binMask1;
    uint16_t binOffset0;
    uint16_t binOffset1;
};

__device__ __forceinline__ uint32_t ExtractDigit(const uint32_t& key, const uint32_t& bit)
{
    return key >> bit & 1;
}

//routine specifically for processing the min size bin
template<uint32_t BITS_TO_SORT>
__device__ __forceinline__ void CuteSort32Bin(
    uint32_t& key,
    uint16_t& index,
    const BinInfo32 binInfo,
    const uint32_t totalLocalLength)
{
    //Finding the exact mask for ballot not worthwhile
    uint32_t eqMask = 0xffffffff;
    uint32_t gtMask = 0;

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

    if (getLaneId() < totalLocalLength)
    {
        index = __popc(gtMask & binInfo.binMask);
        index += __popc(eqMask & binInfo.binMask & getLaneMaskLt());
        index += binInfo.binOffset;
    }
}

template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
__device__ __forceinline__ void CuteSort32(
    uint32_t* keys,
    uint16_t* indexes,
    uint32_t* s_preMerge,
    const uint32_t totalLocalLength,
    const uint32_t warpOffset)
{
    #pragma unroll
    for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
    {
        if (k * LANE_COUNT + warpOffset < totalLocalLength)
        {
            uint32_t eqMask = 0xffffffff;
            uint32_t gtMask = 0;

            #pragma unroll
            for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
            {
                const bool isBitSet = ExtractDigit(keys[k], bit);
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

            if (getLaneId() + k * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k] = __popc(gtMask) + __popc(eqMask & getLaneMaskLt());
                s_preMerge[indexes[k] + k * LANE_COUNT] = keys[k];
            }
            else
            {
                indexes[k] = getLaneId() + k * LANE_COUNT + warpOffset;
            }
        }
        else
        {
            indexes[k] = getLaneId() + k * LANE_COUNT + warpOffset;
        }
    }
}

//Sort 64 keys at a time instead of 32, KEYS_PER_THREAD must be even
template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
__device__ __forceinline__ void CuteSort64(
    uint32_t* keys,
    uint16_t* indexes,
    uint32_t* s_preMerge,
    const uint32_t totalLocalLength,
    const uint32_t warpOffset)
{
    #pragma unroll
    for (uint32_t k = 0; k < KEYS_PER_THREAD; k += 2)
    {
        if (k * LANE_COUNT + warpOffset < totalLocalLength)
        {
            uint64_t eqMask[2] = { ~0ULL, ~0ULL };
            uint64_t gtMask[2] = { 0, 0 };
            uint32_t ballot[2];
            bool setBit[2];

            #pragma unroll
            for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
            {
                setBit[0] = ExtractDigit(keys[k], bit);
                ballot[0] = __ballot_sync(0xffffffff, setBit[0]);

                setBit[1] = ExtractDigit(keys[k + 1], bit);
                ballot[1] = __ballot_sync(0xffffffff, setBit[1]);

                if (setBit[0])
                {
                    eqMask[0] &= reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[0] |= ~reinterpret_cast<uint64_t*>(ballot)[0];
                }
                else
                {
                    eqMask[0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                }

                if (setBit[1])
                {
                    eqMask[1] &= reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[1] |= ~reinterpret_cast<uint64_t*>(ballot)[0];
                }
                else
                {
                    eqMask[1] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[1] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                }
            }

            if (getLaneId() + k * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k] = __popc((uint32_t)eqMask[0] & getLaneMaskLt());
                indexes[k] += __popcll(gtMask[0]);
                s_preMerge[indexes[k] + (k >> 1 << 6)] = keys[k];
            }
            else
            {
                indexes[k] = getLaneId() + k * LANE_COUNT + warpOffset;
            }

            if (getLaneId() + (k + 1) * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k + 1] = __popcll(eqMask[1] & ((uint64_t)getLaneMaskLt() << 32 | 0xffffffff));
                indexes[k + 1] += __popcll(gtMask[1]);
                s_preMerge[indexes[k + 1] + (k >> 1 << 6)] = keys[k + 1];
            }
            else
            {
                indexes[k + 1] = getLaneId() + (k + 1) * LANE_COUNT + warpOffset;
            }
        }
        else
        {
            indexes[k] = getLaneId() + k * LANE_COUNT + warpOffset;
            indexes[k + 1] = getLaneId() + (k + 1) * LANE_COUNT + warpOffset;
        }
    }
}

//Sort 128 keys at a time instead of 32, KEYS_PER_THREAD must be a multiple of 4
template<uint32_t BITS_TO_SORT, uint32_t KEYS_PER_THREAD>
__device__ __forceinline__ void CuteSort128(
    uint32_t* keys,
    uint16_t* indexes,
    uint32_t* s_preMerge,
    const uint32_t totalLocalLength,
    const uint32_t warpOffset)
{
    #pragma unroll
    for (uint32_t k = 0; k < KEYS_PER_THREAD; k += 4)
    {
        if (k * LANE_COUNT + warpOffset < totalLocalLength)
        {
            uint64_t eqMask[4][2] = { {~0ULL, ~0ULL}, {~0ULL, ~0ULL}, {~0ULL, ~0ULL}, {~0ULL, ~0ULL} };
            uint64_t gtMask[4][2] = { {0, 0}, {0, 0}, {0, 0}, {0, 0} };
            uint32_t ballot[4];
            bool setBit[4];

            #pragma unroll
            for (uint32_t bit = 0; bit < BITS_TO_SORT; ++bit)
            {
                setBit[0] = ExtractDigit(keys[k], bit);
                ballot[0] = __ballot_sync(0xffffffff, setBit[0]);
                setBit[1] = ExtractDigit(keys[k + 1], bit);
                ballot[1] = __ballot_sync(0xffffffff, setBit[1]);
                setBit[2] = ExtractDigit(keys[k + 2], bit);
                ballot[2] = __ballot_sync(0xffffffff, setBit[2]);
                setBit[3] = ExtractDigit(keys[k + 3], bit);
                ballot[3] = __ballot_sync(0xffffffff, setBit[3]);

                if (setBit[0])
                {
                    eqMask[0][0] &= reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[0][0] |= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[0][1] |= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }
                else
                {
                    eqMask[0][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[0][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[0][1] &= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }

                if (setBit[1])
                {
                    eqMask[1][0] &= reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[1][0] |= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[1][1] |= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }
                else
                {
                    eqMask[1][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[1][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[1][1] &= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }

                if (setBit[2])
                {
                    eqMask[2][0] &= reinterpret_cast<uint64_t*>(ballot)[0];
                    eqMask[2][1] &= reinterpret_cast<uint64_t*>(ballot)[1];
                    gtMask[2][0] |= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[2][1] |= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }
                else
                {
                    eqMask[2][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    eqMask[2][1] &= ~reinterpret_cast<uint64_t*>(ballot)[1];
                    gtMask[2][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[2][1] &= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }

                if (setBit[3])
                {
                    eqMask[3][0] &= reinterpret_cast<uint64_t*>(ballot)[0];
                    eqMask[3][1] &= reinterpret_cast<uint64_t*>(ballot)[1];
                    gtMask[3][0] |= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[3][1] |= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }
                else
                {
                    eqMask[3][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    eqMask[3][1] &= ~reinterpret_cast<uint64_t*>(ballot)[1];
                    gtMask[3][0] &= ~reinterpret_cast<uint64_t*>(ballot)[0];
                    gtMask[3][1] &= ~reinterpret_cast<uint64_t*>(ballot)[1];
                }
            }

            const uint64_t upperMask = (uint64_t)getLaneMaskLt() << 32ULL | 0xffffffff;

            if (getLaneId() + k * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k] = __popcll(gtMask[0][0]) + __popcll(gtMask[0][1]);
                indexes[k] += __popc((uint32_t)eqMask[0][0] & getLaneMaskLt());
                s_preMerge[indexes[k] + (k >> 2 << 7)] = keys[k];
            }
            else
            {
                indexes[k] = getLaneId() + k * LANE_COUNT + warpOffset;
            }

            if (getLaneId() + (k + 1) * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k + 1] = __popcll(gtMask[1][0]) + __popcll(gtMask[1][1]);
                indexes[k + 1] += __popcll(eqMask[1][0] & upperMask);
                s_preMerge[indexes[k + 1] + (k >> 2 << 7)] = keys[k + 1];
            }
            else
            {
                indexes[k + 1] = getLaneId() + (k + 1) * LANE_COUNT + warpOffset;
            }

            if (getLaneId() + (k + 2) * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k + 2] = __popcll(gtMask[2][0]) + __popcll(gtMask[2][1]);
                indexes[k + 2] += __popcll(eqMask[2][0]) + __popc((uint32_t)eqMask[2][1] & getLaneMaskLt());
                s_preMerge[indexes[k + 2] + (k >> 2 << 7)] = keys[k + 2];
            }
            else
            {
                indexes[k + 2] = getLaneId() + (k + 2) * LANE_COUNT + warpOffset;
            }

            if (getLaneId() + (k + 3) * LANE_COUNT + warpOffset < totalLocalLength)
            {
                indexes[k + 3] = __popcll(gtMask[3][0]) + __popcll(gtMask[3][1]);
                indexes[k + 3] += __popcll(eqMask[3][0]) + __popcll(eqMask[3][1] & upperMask);
                s_preMerge[indexes[k + 3] + (k >> 2 << 7)] = keys[k + 3];
            }
            else
            {
                indexes[k + 3] = getLaneId() + (k + 3) * LANE_COUNT + warpOffset;
            }
        }
        else
        {
            indexes[k] = getLaneId() + k * LANE_COUNT + warpOffset;
            indexes[k + 1] = getLaneId() + (k + 1) * LANE_COUNT + warpOffset;
            indexes[k + 2] = getLaneId() + (k + 2) * LANE_COUNT + warpOffset;
            indexes[k + 3] = getLaneId() + (k + 3) * LANE_COUNT + warpOffset;
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

//Binary search for the segment in which this lane belongs to
__device__ __forceinline__ uint2 BinarySearch(
    const uint32_t* s_warpHist,
    const int32_t binCount)
{
    const uint32_t start = s_warpHist[0];
    if (binCount > 1)
    {
        const uint32_t t = start + getLaneId();
        int32_t l = 0;
        int32_t h = binCount;

        while (l < h)
        {
            const int32_t m = l + (h - l) / 2;
            if (m >= binCount)  //Unnecessary?
                break;
                
            const uint32_t lr = s_warpHist[m];
            const uint32_t rr = s_warpHist[m + 1];
            if (lr <= t && t < rr)
                return { lr - start, rr - start };
            else if (t < lr)
                h = m;
            else
                l = m + 1;
        }

        return { 0, 0 };
    }
    else
    {
        return { 0, s_warpHist[1] - start };
    }
}

__device__ __forceinline__ BinInfo32 GetBinInfo32(
    const uint32_t* s_warpBins,
    const uint32_t binCount,
    const uint32_t binOffset)
{
    const uint2 interval = BinarySearch(s_warpBins, (int32_t)binCount);
    const uint32_t binMask = (((1ULL << interval.y) - 1) >> interval.x << interval.x);
    return BinInfo32{ binMask, interval.x };
}

// Only a single warp participates in sorting a run of keys
// However, multiple independent warps can be launched in a single block
template<
    uint32_t WARP_KEYS,
    uint32_t BLOCK_KEYS,
    uint32_t WARPS,
    uint32_t BITS_TO_SORT>
__device__ __forceinline__ void SplitSortBins32(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    const uint32_t* minBinSegCounts,
    uint32_t* sort,
    uint32_t* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    __shared__ uint32_t s_bins[BLOCK_KEYS];
    uint32_t* s_warpBins = &s_bins[WARP_INDEX * WARP_KEYS];

    const uint32_t binOffset = binOffsets[blockIdx.x * WARPS + WARP_INDEX];
    const uint32_t binCount = minBinSegCounts[blockIdx.x * WARPS + WARP_INDEX];
    LoadBins(segments, s_warpBins, binCount, binOffset, totalSegCount, totalSegLength);
    const BinInfo32 binInfo = GetBinInfo32(s_warpBins, binCount, binOffset);

    const uint32_t segmentStart = s_warpBins[0];
    const uint32_t totalLocalLength = s_warpBins[binCount] - segmentStart;

    uint32_t key = getLaneId() < totalLocalLength ? sort[getLaneId() + segmentStart] : 0xffffffff;

    uint16_t index;
    CuteSort32Bin<BITS_TO_SORT>(key, index, binInfo, totalLocalLength);

    uint32_t payload;
    if (getLaneId() < totalLocalLength)
    {
        sort[index + segmentStart] = key;
        payload = payloads[getLaneId() + segmentStart];
    }
    __syncwarp(0xffffffff);

    if (getLaneId() < totalLocalLength)
        payloads[index + segmentStart] = payload;
}

__device__ __forceinline__ uint2 GetDiags(
    uint32_t* a,
    uint32_t* b,
    const int32_t lengthA,
    const int32_t lengthB,
    const int32_t id,
    const int32_t threadCount)
{
    const int32_t index = id * (lengthA + lengthB) / threadCount;
    int32_t aTop;
    int32_t bTop;
    int32_t aBot;

    if (index > lengthA)
    {
        aTop = lengthA;
        bTop = index - lengthA;
    }
    else
    {
        aTop = index;
        bTop = 0;
    }
    aBot = bTop;

    int32_t aI;
    int32_t bI;
    while (true)
    {
        const int32_t offset = abs(aTop - aBot) / 2;
        aI = aTop - offset;
        bI = bTop + offset;

        if (aI >= 0 && bI <= lengthB && (bI == 0 || aI == lengthA || a[aI] > b[bI - 1]))
        {
            if (aI == 0 || bI == lengthB || a[aI - 1] <= b[bI])
            {
                return uint2{ (uint32_t)aI, (uint32_t)bI };
            }
            else
            {
                aTop = aI - 1;
                bTop = bI + 1;
            }
        }
        else
        {
            aBot = aI + 1;
        }
    }
    return uint2{ (uint32_t)aI, (uint32_t)bI };
}

//For merge of 32 x 32 only
//Gets diagonals but using registers and shuffling as opposed to shared memory
__device__ __forceinline__ uint2 RegGetDiags(
    uint32_t* a,
    uint32_t* b,
    uint32_t id)
{
    const uint32_t index = id << 1;
    uint32_t aTop;
    uint32_t bTop;
    uint32_t aBot;
    bool incomplete = true;

    if (index > LANE_COUNT)
    {
        aTop = LANE_COUNT;
        bTop = index - LANE_COUNT;
    }
    else
    {
        aTop = index;
        bTop = 0;
    }
    aBot = bTop;

    uint2 diags;
    #pragma unroll
    for (uint32_t i = 0; i < LANE_LOG + 1; ++i)
    {
        const uint32_t offset = abs((int32_t)aTop - (int32_t)aBot) >> 1;
        diags = { aTop - offset, bTop + offset };

        const uint32_t xVal = __shfl_sync(0xffffffff, a[0], diags.x == LANE_COUNT ? 0 : diags.x);
        const uint32_t yVal = __shfl_sync(0xffffffff, b[0], diags.y == 0 ? 0 : diags.y - 1);
        const uint32_t xVal2 = __shfl_sync(0xffffffff, a[0], diags.x == 0 ? 0 : diags.x - 1);
        const uint32_t yVal2 = __shfl_sync(0xffffffff, b[0], diags.y == LANE_COUNT ? 0 : diags.y);

        if (incomplete)
        {
            if (diags.y == 0 || diags.x == LANE_COUNT || xVal > yVal)
            {
                if (diags.x == 0 || diags.y == LANE_COUNT || xVal2 <= yVal2)
                {
                    incomplete = false;
                }
                else
                {
                    aTop = diags.x - 1;
                    bTop = diags.y + 1;
                }
            }
            else
            {
                aBot = diags.x + 1;
            }
        }
    }
    return diags;
}

__device__ __forceinline__ void MergeSwapIndex(
    uint32_t* dest,
    uint32_t* source,
    uint32_t destIndex,
    uint32_t sourceIndex)
{
    dest[destIndex] = source[sourceIndex];
    source[sourceIndex] = destIndex;
}

__device__ __forceinline__ void Merge(
    uint32_t* a,
    uint32_t* b,
    uint32_t* c,
    uint32_t startA,
    const uint32_t endA,
    uint32_t startB,
    const uint32_t endB)
{
    if (startA >= endA)     //>= makes code more resiliant
    {
        for (; startB < endB; ++startB)
            MergeSwapIndex(c, b, startA + startB, startB);
    }
    else
    {
        if (startB >= endB) //>= makes code more resiliant
        {
            for (; startA < endA; ++startA)
                MergeSwapIndex(c, a, startA + startB, startA);
        }
        else
        {
            while (true)
            {
                if (a[startA] <= b[startB])
                {
                    MergeSwapIndex(c, a, startA + startB, startA);
                    ++startA;

                    if (startA == endA)
                    {
                        for (; startB < endB; ++startB)
                            MergeSwapIndex(c, b, startA + startB, startB);
                        break;
                    }
                }

                if (a[startA] > b[startB])
                {
                    MergeSwapIndex(c, b, startA + startB, startB);
                    ++startB;

                    if (startB == endB)
                    {
                        for (; startA < endA; ++startA)
                            MergeSwapIndex(c, a, startA + startB, startA);
                        break;
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void GetDiagsAndMergeWarp(
    uint32_t* preMerge,
    uint32_t* postMerge,
    const uint32_t startA,
    const uint32_t attemptMergeLength,
    const uint32_t totalLocalLength,
    const uint32_t warpOffset)
{
    int32_t lengthA = totalLocalLength - (startA + warpOffset);
    if (lengthA < 0)
        lengthA = 0;
    if (lengthA > attemptMergeLength)
        lengthA = attemptMergeLength;

    int32_t lengthB = totalLocalLength - (startA + attemptMergeLength + warpOffset);
    if (lengthB < 0)
        lengthB = 0;
    if (lengthB > attemptMergeLength)
        lengthB = attemptMergeLength;

    const uint2 diags = GetDiags(
        &preMerge[startA],
        &preMerge[startA + attemptMergeLength],
        lengthA,
        lengthB,
        (int32_t)getLaneId(),
        LANE_COUNT);

    uint2 upperDiags{   __shfl_down_sync(0xffffffff, diags.x, 1, LANE_COUNT),
                        __shfl_down_sync(0xffffffff, diags.y, 1, LANE_COUNT) };
    if (getLaneId() == LANE_MASK)
        upperDiags = { (uint32_t)lengthA, (uint32_t)lengthB };
    __syncwarp(0xffffffff);

    /*if (!blockIdx.x)
        printf("%u %u | %u %u | %u\n", diags.x, upperDiags.x, diags.y, upperDiags.y, lengthB);*/

    Merge(
        &preMerge[startA],
        &preMerge[startA + attemptMergeLength],
        &postMerge[startA],
        diags.x,
        upperDiags.x,
        diags.y,
        upperDiags.y);
    __syncwarp(0xffffffff);
}

template<uint32_t START_LOG, uint32_t END_LOG, uint32_t KEYS_PER_THREAD>
__device__ __forceinline__ void MultiLevelMergeWarp(
    uint32_t* preMerge,
    uint32_t* postMerge,
    uint16_t* indexes,
    const uint32_t totalLocalLength,
    const uint32_t warpOffset)
{
    #pragma unroll
    for (uint32_t m = START_LOG; m < END_LOG; ++m)
    {
        if ((m & 1) == (START_LOG & 1))  //flip flop the pre and post merge arrays
        {
            #pragma unroll
            for (uint32_t i = 0; i < (1 << END_LOG - m); i += 2)
                GetDiagsAndMergeWarp(preMerge, postMerge, i << m, 1 << m, totalLocalLength, warpOffset);

            #pragma unroll
            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
            {
                if(indexes[k] + warpOffset < totalLocalLength)
                    indexes[k] = preMerge[indexes[k] + (k >> m - LANE_LOG << m)];
            }
        }
        else
        {
            #pragma unroll
            for (uint32_t i = 0; i < (1 << END_LOG - m); i += 2)
                GetDiagsAndMergeWarp(postMerge, preMerge, i << m, 1 << m, totalLocalLength, warpOffset);

            #pragma unroll
            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
            {
                if (indexes[k] + warpOffset < totalLocalLength)
                    indexes[k] = postMerge[indexes[k] + (k >> m - LANE_LOG << m)];
            }
        }
    }
}

template<
    uint32_t KEYS_PER_THREAD,
    uint32_t WARP_KEYS,
    uint32_t BLOCK_KEYS,
    uint32_t WARPS,
    uint32_t WARP_LOG_START,
    uint32_t WARP_LOG_END>
__device__ __forceinline__ void SplitSortWarp(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    uint32_t* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength,
    void (*CuteSortVariant)(uint32_t*, uint16_t*, uint32_t*, const uint32_t, const uint32_t))
{
    __shared__ uint32_t s_memPreMerge[BLOCK_KEYS];
    __shared__ uint32_t s_memPostMerge[BLOCK_KEYS];

    uint32_t* s_warpMemPreMerge = &s_memPreMerge[WARP_INDEX * WARP_KEYS];
    uint32_t* s_warpMemPostMerge = &s_memPostMerge[WARP_INDEX * WARP_KEYS];

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

    uint16_t indexes[KEYS_PER_THREAD];
    (*CuteSortVariant)(keys, indexes, s_warpMemPreMerge, totalLocalLength, 0);
    __syncwarp(0xffffffff);

    MultiLevelMergeWarp<
        WARP_LOG_START,
        WARP_LOG_END,
        KEYS_PER_THREAD>(
            s_warpMemPreMerge,
            s_warpMemPostMerge,
            indexes,
            totalLocalLength,
            0);

    if (!(WARP_LOG_END - WARP_LOG_START & 1))
    {
        uint32_t* t = s_warpMemPostMerge;
        s_warpMemPostMerge = s_warpMemPreMerge;
        s_warpMemPreMerge = t;
    }

    #pragma unroll
    for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
    {
        if(i < totalLocalLength)
            sort[i + segmentStart] = s_warpMemPostMerge[i];
    }

    uint32_t vals[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
    {
        if (i < totalLocalLength)
            vals[k] = payloads[i + segmentStart];
    }
    __syncwarp(0xffffffff);

    #pragma unroll
    for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
        s_warpMemPreMerge[indexes[k]] = vals[k];          //No check necessary
    __syncwarp(0xffffffff);

    #pragma unroll
    for (uint32_t i = getLaneId(), k = 0; k < KEYS_PER_THREAD; i += LANE_COUNT, ++k)
    {
        if (i < totalLocalLength)
            payloads[i + segmentStart] = s_warpMemPreMerge[i];
    }
}

__device__ __forceinline__ void GetDiagsAndMergeBlock(
    uint32_t* preMerge,
    uint32_t* postMerge,
    const uint32_t startA,
    const uint32_t attemptMergeLength,
    const uint32_t maxWarp,
    const uint32_t totalLocalLength)
{
    int32_t lengthA = totalLocalLength - startA;
    if (lengthA < 0)
        lengthA = 0;
    if (lengthA > attemptMergeLength)
        lengthA = attemptMergeLength;

    int32_t lengthB = totalLocalLength - (startA + attemptMergeLength);
    if (lengthB < 0)
        lengthB = 0;
    if (lengthB > attemptMergeLength)
        lengthB = attemptMergeLength;

    const uint2 diags = GetDiags(
        &preMerge[startA],
        &preMerge[startA + attemptMergeLength],
        lengthA,
        lengthB,
        (int32_t)(threadIdx.x & (maxWarp << LANE_LOG) - 1),
        (int32_t)(maxWarp << LANE_LOG));

    __shared__ uint2 s_upperDiag[8]; //TODO improve this . .. or something :)
    uint2 upperDiags{   __shfl_down_sync(0xffffffff, diags.x, 1, LANE_COUNT),
                        __shfl_down_sync(0xffffffff, diags.y, 1, LANE_COUNT) };
    if (!getLaneId() && (WARP_INDEX & maxWarp - 1))
        s_upperDiag[WARP_INDEX - 1] = diags;
    __syncthreads();
    if (getLaneId() == LANE_MASK)
    {
        upperDiags = (WARP_INDEX & maxWarp - 1) == (maxWarp - 1) ?
            uint2{ (uint32_t)lengthA, (uint32_t)lengthB } : s_upperDiag[WARP_INDEX];
    }

    Merge(
        &preMerge[startA],
        &preMerge[startA + attemptMergeLength],
        &postMerge[startA],
        diags.x,
        upperDiags.x,
        diags.y,
        upperDiags.y);
    __syncthreads();
}

//number of participating warps is implied by the difference between starting and ending log
template<uint32_t START_LOG, uint32_t END_LOG, uint32_t KEYS_PER_THREAD>
__device__ __forceinline__ void MultiLevelMergeBlock(
    uint32_t* preMerge,
    uint32_t* postMerge,
    uint16_t* index,
    const uint32_t totalLocalLength)
{
    #pragma unroll
    for (uint32_t m = START_LOG, w = 1; m < END_LOG; ++m, ++w)
    {
        if ((m & 1) == (START_LOG & 1))  //flip flop the pre and post merge arrays
        {
            GetDiagsAndMergeBlock(preMerge, postMerge, WARP_INDEX >> w << m + 1, 1 << m, 1 << w, totalLocalLength);

            #pragma unroll
            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
            {
                if (index[k] < totalLocalLength)
                    index[k] = preMerge[index[k] + (WARP_INDEX >> w - 1 << m)];
            }
        }
        else
        {
            GetDiagsAndMergeBlock(postMerge, preMerge, WARP_INDEX >> w << m + 1, 1 << m, 1 << w, totalLocalLength);

            #pragma unroll
            for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
            {
                if (index[k] < totalLocalLength)
                    index[k] = postMerge[index[k] + (WARP_INDEX >> w - 1 << m)];
            }
        }
        __syncthreads();
    }
}

template<
    uint32_t KEYS_PER_THREAD,
    uint32_t WARP_KEYS,
    uint32_t BLOCK_KEYS,
    uint32_t WARP_GROUPS,
    uint32_t WARPS_PER_WARP_GROUP,
    uint32_t WARP_LOG_START,
    uint32_t WARP_LOG_END,
    uint32_t BLOCK_LOG_END>
__device__ __forceinline__ void SplitSortBlock(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    uint32_t* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength,
    void (*CuteSortVariant)(uint32_t*, uint16_t*, uint32_t*, const uint32_t, const uint32_t))
{
    __shared__ uint32_t s_memPreMerge[BLOCK_KEYS];
    __shared__ uint32_t s_memPostMerge[BLOCK_KEYS];

    uint32_t* s_warpMemPreMerge = &s_memPreMerge[WARP_INDEX * WARP_KEYS];
    uint32_t* s_warpMemPostMerge = &s_memPostMerge[WARP_INDEX * WARP_KEYS];

    const uint32_t binOffset = binOffsets[blockIdx.x * WARP_GROUPS + WARP_INDEX / WARPS_PER_WARP_GROUP];
    const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
    const uint32_t segmentStart = segments[binOffset];
    const uint32_t totalLocalLength = segmentEnd - segmentStart;

    uint32_t keys[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX % WARPS_PER_WARP_GROUP * WARP_KEYS, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        keys[k] = i < totalLocalLength ? sort[i + segmentStart] : 0xffffffff;
    }

    uint16_t index[KEYS_PER_THREAD];
    (*CuteSortVariant)(keys, index, s_warpMemPreMerge, totalLocalLength, WARP_INDEX % WARPS_PER_WARP_GROUP * WARP_KEYS);
    __syncwarp(0xffffffff);

    MultiLevelMergeWarp<
        WARP_LOG_START,
        WARP_LOG_END,
        KEYS_PER_THREAD>(
            s_warpMemPreMerge,
            s_warpMemPostMerge,
            index,
            totalLocalLength,
            WARP_INDEX % WARPS_PER_WARP_GROUP * WARP_KEYS);
    __syncthreads();
    
    uint32_t* s_warpGroupPreMerge;
    uint32_t* s_warpGroupPostMerge;
    if (WARP_LOG_END - WARP_LOG_START & 1)
    {
        s_warpGroupPreMerge = s_memPostMerge;
        s_warpGroupPostMerge = s_memPreMerge;
    }
    else
    {
        s_warpGroupPreMerge = s_memPreMerge;
        s_warpGroupPostMerge = s_memPostMerge;
    }

    MultiLevelMergeBlock<
        WARP_LOG_END,
        BLOCK_LOG_END,
        KEYS_PER_THREAD>(
            s_warpGroupPreMerge,
            s_warpGroupPostMerge,
            index,
            totalLocalLength);
    
    if (!(BLOCK_LOG_END - WARP_LOG_END & 1))
    {
        uint32_t* t = s_warpGroupPreMerge;
        s_warpGroupPreMerge = s_warpGroupPostMerge;
        s_warpGroupPostMerge = t;
    }

    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX % WARPS_PER_WARP_GROUP * WARP_KEYS, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        if (i < totalLocalLength)
            sort[i + segmentStart] = s_warpGroupPostMerge[i];
    }

    uint32_t vals[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX % WARPS_PER_WARP_GROUP * WARP_KEYS, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        if (i < totalLocalLength)
            vals[k] = payloads[i + segmentStart];
    }

    #pragma unroll
    for (uint32_t k = 0; k < KEYS_PER_THREAD; ++k)
        s_warpGroupPreMerge[index[k]] = vals[k];        //No check necessary
    __syncthreads();

    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX % WARPS_PER_WARP_GROUP * WARP_KEYS, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        if (i < totalLocalLength)
            payloads[i + segmentStart] = s_warpGroupPreMerge[i];
    }
}

#define RADIX_LOG   8
#define RADIX       256
#define RADIX_MASK  255

template<uint32_t KEYS_PER_THREAD, uint32_t KEYS_PER_WARP>
__device__ __forceinline__ void RankKeys(
    uint32_t* keys,
    uint16_t* offsets,
    uint32_t* warpHist,
    const uint32_t radixShift,
    const uint32_t totalLocalLength)
{
    #pragma unroll
    for (uint32_t i = 0; i < KEYS_PER_THREAD; ++i)
    {
        if (i * LANE_COUNT + WARP_INDEX * KEYS_PER_WARP < totalLocalLength)
        {
            uint32_t eqMask = 0xffffffff;
            #pragma unroll
            for (uint32_t bit = radixShift; bit < radixShift + RADIX_LOG; ++bit)
            {
                const bool isBitSet = ExtractDigit(keys[i], bit);
                unsigned ballot = __ballot_sync(0xffffffff, isBitSet);
                eqMask &= isBitSet ? ballot : ~ballot;
            }
            const uint32_t ltEqPeers = __popc(eqMask & getLaneMaskLt());
            uint32_t preIncrementVal;
            if (ltEqPeers == 0)
                preIncrementVal = atomicAdd((uint32_t*)&warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(eqMask));
            offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(eqMask) - 1) + ltEqPeers;
        }
    }
}

template<
    uint32_t WARPS,
    uint32_t KEYS_PER_THREAD,
    uint32_t KEYS_PER_WARP,
    uint32_t PART_SIZE,
    uint32_t BITS_TO_SORT>
__device__ __forceinline__ void SplitSortRadix(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    uint32_t* sort,
    uint32_t* payloads,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength)
{
    __shared__ uint32_t s_hist[RADIX * WARPS];
    __shared__ uint16_t s_indexes[PART_SIZE];

    const uint32_t binOffset = binOffsets[blockIdx.x];
    const uint32_t segmentEnd = binOffset + 1 == totalSegCount ? totalSegLength : segments[binOffset + 1];
    const uint32_t segmentStart = segments[binOffset];
    const uint32_t totalLocalLength = segmentEnd - segmentStart;

    uint32_t* s_warpHist = &s_hist[WARP_INDEX * RADIX];
    for (uint32_t i = getLaneId(); i < RADIX; i += LANE_COUNT)
        s_warpHist[i] = 0;

    uint32_t keys[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        keys[k] = i < totalLocalLength ? sort[i + segmentStart] : 0xffffffff;
    }
    __syncthreads();

    uint16_t offsets[KEYS_PER_THREAD];
    uint16_t indexes[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t radixShift = 0; radixShift < BITS_TO_SORT; radixShift += RADIX_LOG)
    {
        if (radixShift)
        {
            for (uint32_t i = getLaneId(); i < RADIX; i += LANE_COUNT)
                s_warpHist[i] = 0;
            __syncthreads();
        }

        RankKeys<KEYS_PER_THREAD, KEYS_PER_WARP>(keys, offsets, s_warpHist, radixShift, totalLocalLength);
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

        if (WARP_INDEX)
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
                if (i < totalLocalLength)
                {
                    s_hist[offsets[k]] = keys[k];
                    s_indexes[i] = offsets[k];
                }
            }
        }
        else
        {
            #pragma unroll
            for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                k < KEYS_PER_THREAD;
                i += LANE_COUNT, ++k)
            {
                if (i < totalLocalLength)
                {
                    s_hist[offsets[k]] = keys[k];
                    indexes[k] = offsets[k];
                }
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
                    if (i < totalLocalLength)
                    {
                        keys[k] = s_hist[i];
                        indexes[k] = s_indexes[indexes[k]];
                    }
                }
            }
            else
            {
                #pragma unroll
                for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
                    k < KEYS_PER_THREAD;
                    i += LANE_COUNT, ++k)
                {
                    if(i < totalLocalLength)
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

    uint32_t vals[KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        if(i < totalLocalLength)
            vals[k] = payloads[i + segmentStart];
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * KEYS_PER_WARP, k = 0;
        k < KEYS_PER_THREAD;
        i += LANE_COUNT, ++k)
    {
        if (i < totalLocalLength)
            s_hist[s_indexes[indexes[k]]] = vals[k];
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t i = threadIdx.x, k = 0; k < KEYS_PER_THREAD; i += blockDim.x, ++k)
    {
        if(i < totalLocalLength)
            payloads[i + segmentStart] = s_hist[i];
    }
}

//Enclose global kernels in namespace for testing
namespace SplitSortVariants
{
    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv32_cute32_bin(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* minBinSegCounts,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBins32<32, 32 * WARPS, WARPS, BITS_TO_SORT>(
            segments,
            binOffsets,
            minBinSegCounts,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<
        uint32_t WARPS,
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv64_cute32_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortWarp<2, 64, 64 * WARPS, WARPS, 5, 6>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv64_cute64_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortWarp<2, 64, 64 * WARPS, WARPS, 6, 6>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv128_cute32_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortWarp<4, 128, 128 * WARPS, WARPS, 5, 7>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv128_cute64_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortWarp<4, 128, 128 * WARPS, WARPS, 6, 7>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t64_kv128_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<2, 64, 128, WARPS / 2, 2, 5, 6, 7>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t64_kv128_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<2, 64, 128, WARPS / 2, 2, 6, 6, 7>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv256_cute32_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 5, 8>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t32_kv256_cute64_wMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 6, 8>(
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
        uint32_t BITS_TO_SORT>
        __global__ void t32_kv256_cute128_wMerge(
            const uint32_t* segments,
            const uint32_t* binOffsets,
            uint32_t* sort,
            uint32_t* payloads,
            const uint32_t totalSegCount,
            const uint32_t totalSegLength)
    {
        SplitSortWarp<8, 256, WARPS * 256, WARPS, 7, 8>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t64_kv256_cute32_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<4, 128, 256, WARPS / 2, 2, 5, 7, 8>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t64_kv256_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<4, 128, 256, WARPS / 2, 2, 6, 7, 8>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t64_kv512_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<8, 256, 512, WARPS / 2, 2, 6, 8, 9>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t128_kv512_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<4, 128, 512, WARPS / 4, 4, 6, 7, 9>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t256_kv512_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<2, 64, 512, WARPS / 8, 8, 6, 6, 9>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t128_kv1024_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<8, 256, 1024, WARPS / 4, 4, 6, 8, 10>(
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
        uint32_t BITS_TO_SORT>
    __global__ void t256_kv1024_cute64_bMerge(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<4, 128, 1024, WARPS / 8, 8, 6, 7, 10>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    //RADIX SORTS
    #define ROUND_UP_BITS_TO_SORT   ((BITS_TO_SORT >> 3) + ((BITS_TO_SORT & 7) ? 1 : 0) << 3)

    template<uint32_t BITS_TO_SORT>
    __global__ void t64_kv128_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<2, 2, 64, 128, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t64_kv256_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<2, 4, 128, 256, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t64_kv512_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<2, 8, 256, 512, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t128_kv512_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<4, 4, 128, 512, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t128_kv1024_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<4, 8, 256, 1024, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t256_kv1024_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<8, 4, 128, 1024, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t256_kv2048_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<8, 8, 256, 2048, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t512_kv2048_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<16, 4, 128, 2048, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }

    template<uint32_t BITS_TO_SORT>
    __global__ void t512_kv4096_radix(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        uint32_t* payloads,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortRadix<16, 8, 256, 4096, ROUND_UP_BITS_TO_SORT>(
            segments,
            binOffsets,
            sort,
            payloads,
            totalSegCount,
            totalSegLength);
    }
}