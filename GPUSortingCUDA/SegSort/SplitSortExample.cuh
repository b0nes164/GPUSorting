/******************************************************************************
 * GPUSorting
 * SplitSort
 * Experimental Hybrid Radix-Merge based SegmentedSort
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/13/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SplitSort/SplitSort.cuh"

#define CUDA_CHECK(_e, _s)                                                                 \
    if (_e != cudaSuccess) {                                                               \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
    }

//This example will demonstrate how to dispatch SplitSort.
void SplitSortExample() {
    cudaError_t cuda_err;
    
    //We have three segments, `totalSegCount`,
    //with individual segment lengths of 16, 16 and 32
    const uint32_t totalSegCount = 3;
    uint32_t exampleSegments[3] = {16, 16, 32};

    //The total segment length is the sum of
    //all the indivudal segment lengths, 64
    const uint32_t totalSegLength = 64;
    uint32_t exampleKeys[totalSegLength] = {45,  87,  23, 56, 12, 34, 89, 90, 123, 45,  67,  22, 1,
                                            32,  12,  98, 76, 44, 23, 12, 0,  127, 89,  34,  56, 78,
                                            100, 101, 47, 99, 88, 2,  55, 66, 77,  88,  33,  44, 55,
                                            66,  77,  88, 99, 11, 22, 33, 44, 55,  6,   78,  89, 45,
                                            23,  12,  0,  33, 22, 11, 44, 12, 99,  100, 127, 3};

    printf("----------FIRST SEGMENT-----------\n");
    for (uint32_t i = 0; i < 16; ++i) {
        printf("%u, ", exampleKeys[i]);
    }
    printf("\n\n");

    printf("----------SECOND SEGMENT-----------\n");
    for (uint32_t i = 16; i < 32; ++i) {
        printf("%u, ", exampleKeys[i]);
    }
    printf("\n\n");

    printf("----------THIRD SEGMENT-----------\n");
    for (uint32_t i = 32; i < 64; ++i) {
        printf("%u, ", exampleKeys[i]);
    }
    printf("\n\n");

    //SplitSort requires the segment lengths to be
    //in exclusive prefix sum form. Because we have so few segments
    //we'll do this on the CPU side, but typically this would be done on the GPU.
    uint32_t prev = 0;
    for (uint32_t i = 0; i < totalSegCount; ++i) {
        const uint32_t t = exampleSegments[i];
        exampleSegments[i] = prev;
        prev += t;
    }

    //Allocate memory on the GPU and send the data from CPU to GPU
    uint32_t* keys;
    uint32_t* values;
    uint32_t* segments;
    cudaMalloc(&keys, totalSegLength * sizeof(uint32_t));
    cudaMalloc(&values, totalSegLength * sizeof(uint32_t));
    cudaMalloc(&segments, totalSegCount * sizeof(uint32_t));
    cudaMemcpy(keys, exampleKeys, totalSegLength * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(values, exampleKeys, totalSegLength * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(segments, exampleSegments, totalSegCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cuda_err = cudaGetLastError();
    CUDA_CHECK(cuda_err, "Initial malloc");

    //Allocate SplitSort memory
    void* tempMem;
    SplitSortAllocateTempMemory(totalSegLength, totalSegCount, tempMem);
    cuda_err = cudaGetLastError();
    CUDA_CHECK(cuda_err, "Initial malloc");

    SplitSortPairs<32>(segments, keys, values, totalSegCount, totalSegLength, tempMem);

    cudaMemcpy(exampleKeys, keys, totalSegLength * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("------------AFTER SORTING---------------\n");
    printf("\n\n");

    printf("----------FIRST SEGMENT-----------\n");
    for (uint32_t i = 0; i < 16; ++i) {
        printf("%u, ", exampleKeys[i]);
    }
    printf("\n\n");

    printf("----------SECOND SEGMENT-----------\n");
    for (uint32_t i = 16; i < 32; ++i) {
        printf("%u, ", exampleKeys[i]);
    }
    printf("\n\n");

    printf("----------THIRD SEGMENT-----------\n");
    for (uint32_t i = 32; i < 64; ++i) {
        printf("%u, ", exampleKeys[i]);
    }
    printf("\n\n");

    cudaFree(keys);
    cudaFree(values);
    cudaFree(segments);
    cudaFree(tempMem);
}
