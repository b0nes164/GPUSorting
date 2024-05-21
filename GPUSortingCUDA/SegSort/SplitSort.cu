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
#include "SplitSort.cuh"

template<uint32_t BITS_TO_SORT>
__global__ void SplitSort::SortLe32(
    const uint32_t* segments,
    const uint32_t* binOffsets,
    const uint32_t* minBinSegCounts,
    uint32_t* sort,
    uint32_t* payloads)
{
}

template<uint32_t BITS_TO_SORT>
__global__ void SplitSort::SortGt32Le64(
    const uint32_t* segments,
    const uint32_t* binOffset,
    uint32_t* sort,
    uint32_t* payloads)
{
    
}
