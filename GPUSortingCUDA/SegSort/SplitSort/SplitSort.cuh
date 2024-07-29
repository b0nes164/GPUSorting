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
#include "SplitSortBinning.cuh"
#include "SplitSortVariants.cuh"
#include "SplitSortLarge.cuh"

#define SEG_INFO_SIZE           16
#define SEG_HIST_SIZE           (SEG_INFO_SIZE - 1)
#define NEXT_FIT_PART_SIZE      2048
#define ROUND_UP_BITS_TO_SORT   ((BITS_TO_SORT >> 3) + ((BITS_TO_SORT & 7) ? 1 : 0) << 3)                    

namespace SplitSortInternal
{
    //***********************************************************************
    //SORTING VARIANTS
    //***********************************************************************
    //w4_t32_kv32_cute32_bin
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortLe32(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        const uint32_t* minBinSegCounts,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortBins32<32, 128, 4, BITS_TO_SORT, V>(
            segments,
            binOffsets,
            minBinSegCounts,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            segCountInBin);
    };

    //w4_t32_kv64_cute64_wMerge
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt32Le64(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<2, 64, 256, 4, 5, 6, V>(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort32<BITS_TO_SORT, 2>);
    }

    //w2_t32_kv128_cute64_wMerge
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt64Le128(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segCountInBin)
    {
        SplitSortWarp<4, 128, 256, 2, 6, 7, V>(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            segCountInBin,
            CuteSort64<BITS_TO_SORT, 4>);
    }

    //w1_t64_kv256_cute64_bMerge
    //w1_t64_kv256_cute128_bMerge
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt128Le256(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        if constexpr (BITS_TO_SORT > 24)
        {
            SplitSortBlock<4, 128, 256, 2, 6, 7, 8, false, V>(
                segments,
                binOffsets,
                sort,
                values,
                totalSegCount,
                totalSegLength,
                CuteSort64<BITS_TO_SORT, 4>);
        }
        else
        {
            SplitSortBlock<4, 128, 256, 2, 7, 7, 8, false, V>(
                segments,
                binOffsets,
                sort,
                values,
                totalSegCount,
                totalSegLength,
                CuteSort128<BITS_TO_SORT, 4>);
        }
    }

    //w1_t128_kv512_cute128_bMerge
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt256Le512(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortBlock<4, 128, 512, 4, 7, 7, 9, false, V>(
        segments,
        binOffsets,
        sort,
        values,
        totalSegCount,
        totalSegLength,
        CuteSort128<BITS_TO_SORT, 4>);
    }

    //w1_t256_kv1024_cute128_bMerge
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt512Le1024(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        if constexpr (BITS_TO_SORT > 24)
        {
            SplitSortBlock<4, 128, 1024, 8, 7, 7, 10, false, V>(
                segments,
                binOffsets,
                sort,
                values,
                totalSegCount,
                totalSegLength,
                CuteSort128<BITS_TO_SORT, 4>);
        }
        else
        {
            __shared__ uint32_t s_mem[256 * 4 + 1024];  //RADIX * WARPS + PART_SIZE
            uint32_t totalLocalLength;
            GetSegmentInfoRadixFine(
                segments,
                binOffsets,
                sort,
                values,
                totalSegCount,
                totalSegLength,
                totalLocalLength);

            if (totalLocalLength <= 640)
            {
                SplitSortRadixFine<4, 5, 160, 640, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                    s_mem,
                    &s_mem[1024], //RADIX * WARPS
                    sort,
                    values,
                    totalLocalLength);
            }

            if (totalLocalLength > 640 && totalLocalLength <= 768)
            {
                SplitSortRadixFine<4, 6, 192, 768, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                    s_mem,
                    &s_mem[1024], //RADIX * WARPS
                    sort,
                    values,
                    totalLocalLength);
            }

            if (totalLocalLength > 768 && totalLocalLength <= 896)
            {
                SplitSortRadixFine<4, 7, 224, 896, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                    s_mem,
                    &s_mem[1024], //RADIX * WARPS
                    sort,
                    values,
                    totalLocalLength);
            }

            if (totalLocalLength > 896)
            {
                SplitSortRadixFine<4, 8, 256, 1024, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                    s_mem,
                    &s_mem[1024], //RADIX * WARPS
                    sort,
                    values,
                    totalLocalLength);
            }
        }
    }

    //w1_t256_kv2048_radix
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void __launch_bounds__(256) SortGt1024Le2048(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        __shared__ uint32_t s_mem[256 * 8 + 2048];  //RADIX * WARPS + PART_SIZE
        uint32_t totalLocalLength;
        GetSegmentInfoRadixFine(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            totalLocalLength);

        if (totalLocalLength <= 1280)
        {
            SplitSortRadixFine<8, 5, 160, 1280, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 1280 && totalLocalLength <= 1536)
        {
            SplitSortRadixFine<8, 6, 192, 1536, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 1536 && totalLocalLength <= 1792)
        {
            SplitSortRadixFine<8, 7, 224, 1792, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 1792)
        {
            SplitSortRadixFine<8, 8, 256, 2048, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[2048], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }
    }

    //w1_t512_kv4096_radix
    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt2048Le4096(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        __shared__ uint32_t s_mem[256 * 16 + 4096];  //RADIX * WARPS + PART_SIZE
        uint32_t totalLocalLength;
        GetSegmentInfoRadixFine(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            totalLocalLength);

        if (totalLocalLength <= 2560)
        {
            SplitSortRadixFine<16, 5, 160, 2560, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 2560 && totalLocalLength <= 3072)
        {
            SplitSortRadixFine<16, 6, 192, 3072, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 3072 && totalLocalLength <= 3584)
        {
            SplitSortRadixFine<16, 7, 224, 3584, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 3584)
        {
            SplitSortRadixFine<16, 8, 256, 4096, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[4096], //RADIX * WARPS
                sort,
                values,
                totalLocalLength);
        }
    }

    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt4096Le6144(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        __shared__ uint32_t s_mem[6144 * 2];
        uint32_t totalLocalLength;
        GetSegmentInfoRadixFine(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            totalLocalLength);

        if (totalLocalLength <= 4608)
        {
            SplitSortRadixFine<16, 9, 288, 4608, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 4608 && totalLocalLength <= 5120)
        {
            SplitSortRadixFine<16, 10, 320, 5120, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 5120 && totalLocalLength <= 5632)
        {
            SplitSortRadixFine<16, 11, 352, 5632, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 5632)
        {
            SplitSortRadixFine<16, 12, 384, 6144, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                s_mem,
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }
    }

    template<uint32_t BITS_TO_SORT, class V>
    __global__ void SortGt6144Le8192(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        extern __shared__ uint32_t s_mem[];
        uint32_t totalLocalLength;
        GetSegmentInfoRadixFine(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            totalLocalLength);

        //Kernel too large, so we lose some granularity here
        if (totalLocalLength <= 7168)
        {
            SplitSortRadixFine<16, 14, 448, 7168, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                &s_mem[0],
                &s_mem[8192], //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }
        
        if(totalLocalLength > 7168)
        {
            SplitSortRadixFine<16, 16, 512, 8192, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                &s_mem[0],
                &s_mem[8192],  //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }
    }

    template<uint32_t BITS_TO_SORT, uint32_t GRID_STRIDE_LOG, class V>
    __global__ void __launch_bounds__(512) SortGt8192(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        extern __shared__ uint32_t s_mem[];
        uint32_t totalLocalLength;
        GetSegmentInfoRadixMerge<V, GRID_STRIDE_LOG, (1 << GRID_STRIDE_LOG) - 1, 8192>(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength,
            totalLocalLength);

        if (totalLocalLength == 0)
            return;

        //Try to capture some granularity here
        if (totalLocalLength <= 6144)
        {
            SplitSortRadixFine<16, 12, 384, 6144, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                &s_mem[0],
                &s_mem[6144], //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }

        if (totalLocalLength > 6144)
        {
            SplitSortRadixFine<16, 16, 512, 8192, ROUND_UP_BITS_TO_SORT, 256, 255, 8, V>(
                &s_mem[0],
                &s_mem[8192],  //RADIX * WARPS or PART_SIZE
                sort,
                values,
                totalLocalLength);
        }
    }

    template<class V, uint32_t START_LOG, uint32_t END_LOG, uint32_t GRID_STRIDE_LOG>
    __global__ void __launch_bounds__(1024) MergeGt8192(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        volatile uint32_t* index,
        volatile uint32_t* gridLock,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength)
    {
        SplitSortMergeDeviceGrid<V, 8, GRID_STRIDE_LOG, (1 << GRID_STRIDE_LOG) - 1, START_LOG, END_LOG, 10>(
            segments,
            binOffsets,
            sort,
            values,
            index,
            gridLock,
            totalSegCount,
            totalSegLength);
    }

    template<
        class V,
        uint32_t BITS_TO_SORT,
        uint32_t START_LOG,
        uint32_t END_LOG,
        uint32_t GRID_STRIDE_LOG>
    __host__ void DispatchGt8192(
        const uint32_t* segments,
        const uint32_t* binOffsets,
        uint32_t* sort,
        V* values,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t segsInCurBin)
    {
        cudaFuncSetAttribute(
            SplitSortInternal::SortGt8192<BITS_TO_SORT, GRID_STRIDE_LOG, V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            65536);
        SplitSortInternal::SortGt8192<
            BITS_TO_SORT,
            GRID_STRIDE_LOG,
            V>
        <<<(segsInCurBin << GRID_STRIDE_LOG), 512, 65536, 0>>>(
            segments,
            binOffsets,
            sort,
            values,
            totalSegCount,
            totalSegLength);

        uint32_t* gridLock;
        cudaMalloc(&gridLock, (segsInCurBin * 2 + 1) * sizeof(uint32_t));
        cudaMemset(gridLock, 0, (segsInCurBin * 2 + 1) * sizeof(uint32_t));

        SplitSortInternal::MergeGt8192<
            V,
            START_LOG,
            END_LOG,
            GRID_STRIDE_LOG>
        <<<(segsInCurBin << GRID_STRIDE_LOG), 1024>>>(
            segments,
            binOffsets,
            sort,
            values,
            gridLock,
            gridLock + 1,
            totalSegCount,
            totalSegLength);

        cudaFree(gridLock);
    }

    __host__ __forceinline__ uint32_t GetNextFitPartitions(uint32_t totalSegCount)
    {
        return dvrup<NEXT_FIT_PART_SIZE>(totalSegCount);
    }

    __host__ __forceinline__ uint32_t* GetAtomicIndexPointer(void* tempMem)
    {
        return (uint32_t*)tempMem;
    }

    __host__ __forceinline__ uint32_t* GetSegHistDevicePointer(void* tempMem)
    {
        return &((uint32_t*)tempMem)[2];
    }

    __host__ __forceinline__ uint32_t* GetNextFitReductionPointer(void* tempMem)
    {
        return &((uint32_t*)tempMem)[2 + SEG_INFO_SIZE];
    }

    __host__ __forceinline__ uint32_t* GetBinAndCoalesceReductionsPointer(void* tempMem, const uint32_t nextFitPartitions)
    {
        return &((uint32_t*)tempMem)[2 + SEG_INFO_SIZE + nextFitPartitions];
    }

    __host__ __forceinline__ uint32_t* GetPackedSegCountsPointer(void* tempMem, const uint32_t nextFitPartitions)
    {
        return &((uint32_t*)tempMem)[2 + SEG_INFO_SIZE + nextFitPartitions * 2];
    }

    __host__ __forceinline__ uint32_t* GetBinOffsetsPointer(void* tempMem, const uint32_t nextFitPartitions, const uint32_t totalSegCount)
    {
        return &((uint32_t*)tempMem)[2 + SEG_INFO_SIZE + nextFitPartitions * 2 + totalSegCount];
    }

    __host__ __forceinline__ uint32_t* GetLargeSegmentOffsetsPointer(void* tempMem, const uint32_t nextFitPartitions, const uint32_t totalSegCount)
    {
        return &((uint32_t*)tempMem)[2 + SEG_INFO_SIZE + (nextFitPartitions + totalSegCount) * 2];
    }

    //***********************************************************************
    //SPLIT SORT BINNING FUNCTION
    //***********************************************************************
    __host__ void SplitSortBinning(
        uint32_t* segments,
        uint32_t* binOffsets,
        uint32_t* packedSegCounts,
        void* tempMem,
        uint32_t* segInfo,
        const uint32_t totalSegCount,
        const uint32_t totalSegLength,
        const uint32_t nextFitPartitions)
    {
        cudaMemset(tempMem, 0, (2 + SEG_INFO_SIZE + nextFitPartitions * 2) * sizeof(uint32_t));
        cudaDeviceSynchronize();

        const uint32_t k_nextFitBlockDim = 64;
        const uint32_t k_nextFitSPT = 32;
        const uint32_t k_minBinSize = 32;
        uint32_t* segHistDevicePointer = GetSegHistDevicePointer(tempMem);
        
        NextFitBinPacking<
            NEXT_FIT_PART_SIZE,
            k_nextFitBlockDim / LANE_COUNT,
            k_nextFitSPT,
            k_minBinSize,
            SEG_INFO_SIZE>
        <<<nextFitPartitions, k_nextFitBlockDim>>>(
            segments,
            segHistDevicePointer,
            packedSegCounts,
            binOffsets,
            GetAtomicIndexPointer(tempMem),
            GetNextFitReductionPointer(tempMem),
            totalSegCount,
            totalSegLength);

        BinningScan<SEG_HIST_SIZE><<<1, 32>>>(segHistDevicePointer);

        cudaMemcpyAsync(segInfo, GetSegHistDevicePointer(tempMem), SEG_INFO_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        //For now, if any segments of length greater than 65536
        //are encountered, the entire segmented sort is done in place
        if (segInfo[0] - segInfo[SEG_INFO_SIZE - 2] == 0)
        {
            const uint32_t k_binThreads = 256;
            BinSimple<k_minBinSize><<<dvrup<k_binThreads>(totalSegCount), k_binThreads>>>(
                segments,
                segHistDevicePointer,
                binOffsets,
                totalSegCount,
                totalSegLength);
        }
    }
}

//***********************************************************************
//TEMPORARY MEMORY
//***********************************************************************
__host__ void SplitSortAllocateTempMemory(
    const uint32_t totalSegLength,
    const uint32_t totalSegCount,
    void*& tempMem)
{
    const uint32_t nextFitPartitions = SplitSortInternal::GetNextFitPartitions(totalSegCount);
    cudaMalloc(&tempMem,
        (totalSegCount +        //BinOffsets
        totalSegCount +         //Maximum possible size of packed segment counts
        totalSegCount +         //LargeSegmentOffsets
        nextFitPartitions +     //For the single pass scan during NextFitBinPacking
        nextFitPartitions +     //For the single pass scan during BinAndCoalesceLarge
        SEG_INFO_SIZE +         //Size of the segment info array
        2)                      //Atomic bumping indexes for single pass scans
        * sizeof(uint32_t));
    //TODO CUDA ERR CHECK MALLOC
}

__host__ void SplitSortFreeTempMemory(void* tempMem)
{
    cudaDeviceSynchronize();
    //TODO CUDA ERR CHECK MALLOC
    cudaFree(tempMem);
}

//***********************************************************************
//SPLIT SORT MAIN FUNCTION
//***********************************************************************
template<uint32_t BITS_TO_SORT, class V>
__host__ void SplitSortPairs(
    uint32_t* segments,
    uint32_t* sort,
    V* values,
    const uint32_t totalSegCount,
    const uint32_t totalSegLength,
    void* tempMem)
{
    //empirical results indicate cuda stream slower?
    //cudaStream_t stream[SEG_HIST_SIZE];
    //for(uint32_t i = 0; i < SEG_HIST_SIZE; ++i)
    //    cudaStreamCreate(&stream[i]);
    
    //The segInfo contains:
    //0 - 13:   CircularShift inclusive scan of segment bin histogram
    //14:       The totalLength of all segments whose size is greater than 8192 
    uint32_t segInfo[SEG_INFO_SIZE];
    const uint32_t nextFitPartitions = SplitSortInternal::GetNextFitPartitions(totalSegCount);
    uint32_t* packedSegCounts = SplitSortInternal::GetPackedSegCountsPointer(tempMem, nextFitPartitions);
    uint32_t* binOffsets = packedSegCounts + totalSegCount;

    //Bin
    SplitSortInternal::SplitSortBinning(
        segments,
        binOffsets,
        packedSegCounts,
        tempMem,
        segInfo,
        totalSegCount,
        totalSegLength,
        nextFitPartitions);
        
    //For now, if any segments of length greater than 65536
    //are encountered, the entire segmented sort is done in place.
    //In the future, we'll only do this if the majority of the
    //segments are greater than 65536
    //segInfo is in circular shifted inclusive/exclusive form
    if (segInfo[0] - segInfo[SEG_INFO_SIZE - 2] > 0)
    {
        SplitSortInternal::SplitSortLargeInPlace<V, BITS_TO_SORT>(
            segments,
            sort,
            values,
            totalSegCount,
            totalSegLength);
    }
    else
    {
        uint32_t segsInCurBin = segInfo[1];
        if (segsInCurBin)
        {
            SplitSortInternal::SortLe32<BITS_TO_SORT><<<SplitSortInternal::dvrup<4>(segsInCurBin), 128>>>(
                segments,
                binOffsets,
                packedSegCounts,
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }
        
        segsInCurBin = segInfo[2] - segInfo[1];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt32Le64<BITS_TO_SORT><<<SplitSortInternal::dvrup<4>(segsInCurBin), 128>>>(
                segments,
                binOffsets + segInfo[1],
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segInfo[3] - segInfo[2];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt64Le128<BITS_TO_SORT><<<SplitSortInternal::dvrup<2>(segsInCurBin), 64>>>(
                segments,
                binOffsets + segInfo[2],
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segInfo[4] - segInfo[3];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt128Le256<BITS_TO_SORT><<<segsInCurBin, 64>>>(
                segments,
                binOffsets + segInfo[3],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[5] - segInfo[4];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt256Le512<BITS_TO_SORT><<<segsInCurBin, 128>>>(
                segments,
                binOffsets + segInfo[4],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[6] - segInfo[5];
        if (segsInCurBin)
        {
            constexpr uint32_t threads = BITS_TO_SORT > 24 ? 256 : 128;
            SplitSortInternal::SortGt512Le1024<BITS_TO_SORT><<<segsInCurBin, threads>>>(
                segments,
                binOffsets + segInfo[5],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[7] - segInfo[6];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt1024Le2048<BITS_TO_SORT><<<segsInCurBin, 256>>>(
                segments,
                binOffsets + segInfo[6],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[8] - segInfo[7];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt2048Le4096<BITS_TO_SORT><<<segsInCurBin, 512>>>(
                segments,
                binOffsets + segInfo[7],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[9] - segInfo[8];
        if (segsInCurBin)
        {
            SplitSortInternal::SortGt4096Le6144<BITS_TO_SORT><<<segsInCurBin, 512>>>(
                segments,
                binOffsets + segInfo[8],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[10] - segInfo[9];
        if (segsInCurBin)
        {
            cudaFuncSetAttribute(
                SplitSortInternal::SortGt6144Le8192<BITS_TO_SORT, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                65536);
            SplitSortInternal::SortGt6144Le8192<BITS_TO_SORT, V><<<segsInCurBin, 512, 65536, 0>>>(
                segments,
                binOffsets + segInfo[9],
                sort,
                values,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segInfo[11] - segInfo[10];
        if (segsInCurBin)
        {
            SplitSortInternal::DispatchGt8192<V, BITS_TO_SORT, 13, 14, 1>(
                segments,
                binOffsets + segInfo[10],
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segInfo[12] - segInfo[11];
        if (segsInCurBin)
        {
            SplitSortInternal::DispatchGt8192<V, BITS_TO_SORT, 13, 15, 2>(
                segments,
                binOffsets + segInfo[11],
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segInfo[13] - segInfo[12];
        if (segsInCurBin)
        {
            SplitSortInternal::DispatchGt8192<V, BITS_TO_SORT, 13, 16, 3>(
                segments,
                binOffsets + segInfo[12],
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        segsInCurBin = segInfo[14] - segInfo[13];
        if (segsInCurBin)
        {
            SplitSortInternal::DispatchGt8192<V, BITS_TO_SORT, 13, 17, 4>(
                segments,
                binOffsets + segInfo[13],
                sort,
                values,
                totalSegCount,
                totalSegLength,
                segsInCurBin);
        }

        //In the future we will use the out of place method
    }
};

#undef ROUND_UP_BITS_TO_SORT
#undef NEXT_FIT_PART_SIZE
#undef SEG_HIST_SIZE
#undef SEG_INFO_SIZE