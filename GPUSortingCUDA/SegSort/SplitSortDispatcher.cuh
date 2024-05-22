/******************************************************************************
*  GPUSorting
 * SplitSort
 * Experimental SegSort that does not use cooperative groups
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 5/16/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SplitSortBinning.cuh"
#include "SplitSort.cuh"
#include "../UtilityKernels.cuh"

class SplitSortDispatcher
{
    const uint32_t k_nextFitSize = 2048;
    const uint32_t k_nextFitThreads = 64;
    const uint32_t k_segHistSize = 9;

    const uint32_t k_maxSize;
    const uint32_t k_maxSegments;

    uint32_t* m_sort;
    uint32_t* m_payloads;
    uint32_t* m_segments;
    uint32_t* m_minBinSegCounts;
    uint32_t* m_binOffsets;
    uint32_t* m_reduction;
    uint32_t* m_segHist;
    uint32_t* m_index;
    uint32_t* m_totalLength;
    uint32_t* m_errCount;

public:
    SplitSortDispatcher(
        uint32_t maxSize,
        uint32_t maxSegments) :
        k_maxSize(maxSize),
        k_maxSegments(maxSegments)
    {
        cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_payloads, k_maxSize * sizeof(uint32_t));
        cudaMalloc(&m_segments, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_minBinSegCounts, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_binOffsets, k_maxSegments * sizeof(uint32_t));
        cudaMalloc(&m_reduction, divRoundUp(k_maxSegments, k_nextFitSize) * sizeof(uint32_t));
        cudaMalloc(&m_segHist, k_segHistSize * sizeof(uint32_t));
        cudaMalloc(&m_index, sizeof(uint32_t));
        cudaMalloc(&m_totalLength, sizeof(uint32_t));
        cudaMalloc(&m_errCount, sizeof(uint32_t));
    }

    ~SplitSortDispatcher()
    {
        cudaFree(m_sort);
        cudaFree(m_payloads);
        cudaFree(m_segments);
        cudaFree(m_minBinSegCounts);
        cudaFree(m_binOffsets);
        cudaFree(m_reduction);
        cudaFree(m_segHist);
        cudaFree(m_index);
        cudaFree(m_totalLength);
        cudaFree(m_errCount);
    }

    void BatchTimeFixedSegmentLength(
        uint32_t batchCount,
        uint32_t totalSegCount,
        uint32_t segLength)
    {
        if (totalSegCount > k_maxSegments)
        {
            printf("Error seg count exceed max allocated memory. \n");
            return;
        }

        if (totalSegCount * segLength > k_maxSize)
        {
            printf("Error sort size exceeds max allocated memory. \n");
            return;
        }

        printf("Beginning Split Sort Pairs Fixed Seg Length Batch Timing: \n");
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        for (uint32_t i = 0; i <= batchCount; ++i)
        {
            InitSegLengthsFixed<<<256,256>>>(m_segments, totalSegCount, segLength);
            InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_sort, segLength, totalSegCount);
            InitFixedSegLengthDescendingValue<<<1024, 64>>>(m_payloads, segLength, totalSegCount);
            //InitFixedSegLengthRandomValue<<<1024,64>>>(m_sort, segLength, totalSegCount, i + 10);
            //InitFixedSegLengthRandomValue<<<1024,64>>>(m_payloads, segLength, totalSegCount, i + 10);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            DispatchSplitSort<32>(totalSegCount, totalSegCount * segLength);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis;
            cudaEventElapsedTime(&millis, start, stop);
            if (i)
                totalTime += millis;

            if ((i & 15) == 0)
                printf(". ");
        }

        totalTime /= 1000.0f;
        uint32_t size = totalSegCount * segLength;

        printf("\n");
        printf("Total time elapsed: %f\n", totalTime);
        printf("Estimated speed at %u 32-bit elements: %E pairs/sec\n\n", size, size / totalTime * batchCount);
    }

private:
    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    template<uint32_t BITS_TO_SORT>
    void DispatchSplitSort(uint32_t totalSegCount, uint32_t totalSegLength)
    {
        uint32_t segHist[9];
        cudaStream_t streams[9 - 1];
        for (uint32_t i = 0; i < k_segHistSize - 1; ++i)
            cudaStreamCreate(&streams[i]);

        const uint32_t binPackPartitions = divRoundUp(totalSegCount, k_nextFitSize);
        cudaMemset(m_index, 0, sizeof(uint32_t));
        cudaMemset(m_segHist, 0, k_segHistSize * sizeof(uint32_t));
        cudaMemset(m_reduction, 0, binPackPartitions * sizeof(uint32_t));
        cudaDeviceSynchronize();

        SplitSortBinning::NextFitBinPacking<<<binPackPartitions, k_nextFitThreads>>>(
            m_segments,
            m_segHist,
            m_minBinSegCounts,
            m_binOffsets,
            m_index,
            m_reduction,
            totalSegCount,
            totalSegLength);

        SplitSortBinning::Scan<<<1, 32>>> (m_segHist);

        cudaMemcpyAsync(segHist, m_segHist, k_segHistSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        SplitSortBinning::Bin<<<divRoundUp(totalSegCount, 256), 256>>>(
            m_segments,
            m_segHist,
            m_binOffsets,
            totalSegCount,
            totalSegLength);

        uint32_t segsInCurBin = segHist[1] - segHist[0];
        if (segsInCurBin)
        {
            SplitSort::SortLe32<BITS_TO_SORT><<<divRoundUp(segsInCurBin, 4), 128>>>(
                m_segments,
                m_binOffsets,
                m_minBinSegCounts,
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }
        
        segsInCurBin = segHist[2] - segHist[1];
        if (segsInCurBin)
        {
            SplitSort::SortGt32Le64<BITS_TO_SORT><<<divRoundUp(segsInCurBin, 4), 128, 0, streams[1]>>>(
                m_segments,
                m_binOffsets + segHist[1],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[3] - segHist[2];
        if (segsInCurBin)
        {
            SplitSort::SortGt64Le128<BITS_TO_SORT><<<segsInCurBin, 64, 0, streams[2]>>>(
                m_segments,
                m_binOffsets + segHist[2],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[4] - segHist[3];
        if (segsInCurBin)
        {
            SplitSort::SortGt128Le256<BITS_TO_SORT><<<segsInCurBin, 128, 0, streams[3]>>>(
                m_segments,
                m_binOffsets + segHist[3],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[5] - segHist[4];
        if (segsInCurBin)
        {
            SplitSort::SortGt256Le512<BITS_TO_SORT><<<segsInCurBin, 128, 0, streams[4]>>>(
                m_segments,
                m_binOffsets + segHist[4],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[6] - segHist[5];
        if (segsInCurBin)
        {
            SplitSort::SortGt512Le1024<BITS_TO_SORT><<<segsInCurBin, 128, 0, streams[5]>>>(
                m_segments,
                m_binOffsets + segHist[5],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[7] - segHist[6];
        if (segsInCurBin)
        {
            SplitSort::SortGt1024Le2048<BITS_TO_SORT><<<segsInCurBin, 256, 0, streams[6]>>>(
                m_segments,
                m_binOffsets + segHist[6],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        segsInCurBin = segHist[8] - segHist[7];
        if (segsInCurBin)
        {
            SplitSort::SortGt2048Le4096<BITS_TO_SORT><<<segsInCurBin, 512, 0, streams[7]>>>(
                m_segments,
                m_binOffsets + segHist[7],
                m_sort,
                m_payloads,
                totalSegCount,
                totalSegLength);
        }

        //onesweep here :)

        for (uint32_t i = 0; i < k_segHistSize - 1; ++i)
            cudaStreamDestroy(streams[i]);
    }
};