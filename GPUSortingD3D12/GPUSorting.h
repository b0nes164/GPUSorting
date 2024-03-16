/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/13/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include "pch.h"

struct DeviceInfo
{
    std::wstring Description;
    std::wstring SupportedShaderModel;
    uint32_t SIMDWidth;
    uint32_t SIMDLaneCount;
    uint32_t SIMDMaxWidth;
    bool SupportsWaveIntrinsics;
    bool Supports16BitTypes;
    bool SupportsDeviceRadixSort;
    bool SupportsOneSweep;
};

typedef 
enum GPU_SORTING_MODE
{
    GPU_SORTING_KEYS_ONLY   = 0,
    GPU_SORTING_PAIRS       = 1,
}   GPU_SORTING_MODE;

typedef
enum GPU_SORTING_ORDER
{
    GPU_SORTING_ASCENDING   = 0,
    GPU_SORTING_DESCENDING  = 1,  
}   GPU_SORTING_ORDER;

typedef
enum GPU_SORTING_KEY_TYPE
{
    GPU_SORTING_KEY_UINT32  = 0,
    GPU_SORTING_KEY_INT32   = 1,
    GPU_SORTING_KEY_FLOAT32 = 2,
}   GPU_SORTING_KEY_TYPE;

typedef
enum GPU_SORTING_PAYLOAD_TYPE
{
    GPU_SORTING_PAYLOAD_UINT32  = 0,
    GPU_SORTING_PAYLOAD_INT32   = 1,
    GPU_SORTING_PAYLOAD_FLOAT32 = 2,
}   GPU_SORTING_PAYLOAD_TYPE;

struct GPUSortingConfig
{
    GPU_SORTING_MODE sortingMode;
    GPU_SORTING_ORDER sortingOrder;
    GPU_SORTING_KEY_TYPE sortingKeyType;
    GPU_SORTING_PAYLOAD_TYPE sortingPayloadType;
};

typedef
enum ENTROPY_PRESET
{
    ENTROPY_PRESET_1 = 0,
    ENTROPY_PRESET_2 = 1,
    ENTROPY_PRESET_3 = 2,
    ENTROPY_PRESET_4 = 3,
    ENTROPY_PRESET_5 = 4,
}   ENTROPY_PRESET;