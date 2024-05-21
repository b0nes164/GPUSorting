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
#include "SplitSort.cuh"
#include "SplitSortBinning.cuh"
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
        cudaMalloc(&m_segHist, 20 * sizeof(uint32_t));
        cudaMalloc(&m_index, sizeof(uint32_t));
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
        cudaFree(m_errCount);
    }

    void Dispatch()
    {
        ValidateBinningPartials();
    }

private:
    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    void DispatchBinning(uint32_t totalSegCount, uint32_t totalSegLength)
    {
        const uint32_t partitions = divRoundUp(totalSegCount, k_nextFitSize);
        cudaMemset(m_index, 0, sizeof(uint32_t));
        cudaMemset(m_segHist, 0, k_segHistSize * sizeof(uint32_t));
        cudaMemset(m_reduction, 0, partitions * sizeof(uint32_t));
        cudaDeviceSynchronize();

        SplitSortBinning::NextFitBinPacking<<<partitions, k_nextFitThreads>>>(
            m_segments,
            m_segHist,
            m_minBinSegCounts,
            m_binOffsets,
            m_index,
            m_reduction,
            totalSegCount,
            totalSegLength);

        SplitSortBinning::Scan<<<1,32>>>(m_segHist);

        SplitSortBinning::Bin<<<divRoundUp(totalSegCount,256), 256>>>(
            m_segments,
            m_segHist,
            m_binOffsets,
            totalSegCount,
            totalSegLength);
    }

    //Super simple validation of the loading of partial tiles
    void ValidateBinningPartials()
    {
        uint32_t validate[9];
        uint32_t testsPassed = 0;

        const uint32_t segLength = 32;
        const uint32_t end = (1 << 19) + k_nextFitSize;
        for (uint32_t i = 1 << 19; i < end; ++i)
        {
            InitSegLengthsFixed<<<256, 256>>>(m_segments, i, segLength);
            DispatchBinning(i, segLength * i);
            
            cudaDeviceSynchronize();
            cudaMemcpy(&validate, m_segHist, k_segHistSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            bool isValid = true;
            if (validate[0])
                isValid = false;
                
            if (validate[1] != i)
                isValid = false;
                
            if (!isValid)
                printf("\nTest failed at size %u \n", i);
            else
                testsPassed++;

            if (!(i & 255))
                printf(". ");
        }
        
        if (testsPassed == k_nextFitSize)
            printf("\nSimple partial validation passed\n");
        else
            printf("\nSimple partial failed with %u errors\n", k_nextFitSize - testsPassed);
    }
};