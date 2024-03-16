/******************************************************************************
 * GPUSorting
 * OneSweep Implementation
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"

namespace OneSweep
{
    __global__ void GlobalHistogram(
        uint32_t* sort,
        uint32_t* globalHistogram,
        uint32_t size);

    __global__ void Scan(
        uint32_t* globalHistogram,
        uint32_t* firstPassHistogram,
        uint32_t* secPassHistogram,
        uint32_t* thirdPassHistogram,
        uint32_t* fourthPassHistogram);

    __global__ void DigitBinningPassKeysOnly(
        uint32_t* sort,
        uint32_t* alt,
        volatile uint32_t* passHistogram,
        volatile uint32_t* index,
        uint32_t size,
        uint32_t radixShift);

    __global__ void DigitBinningPassPairs(
        uint32_t* sort,
        uint32_t* sortPayload,
        uint32_t* alt,
        uint32_t* altPayload,
        volatile uint32_t* passHistogram,
        volatile uint32_t* index,
        uint32_t size,
        uint32_t radixShift);
}