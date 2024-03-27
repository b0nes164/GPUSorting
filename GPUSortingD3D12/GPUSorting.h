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

namespace GPUSorting
{
    struct DeviceInfo
    {
        std::wstring Description;
        std::wstring SupportedShaderModel;
        uint32_t deviceId;
        uint32_t vendorId;
        uint32_t SIMDWidth;
        uint32_t SIMDLaneCount;
        uint32_t SIMDMaxWidth;
        uint64_t dedicatedVideoMemory;
        uint64_t sharedSystemMemory;
        bool SupportsWaveIntrinsics;
        bool Supports16BitTypes;
        bool SupportsDeviceRadixSort;
        bool SupportsOneSweep;
    };

    struct TuningParameters
    {
        bool shouldLockWavesTo32;
        uint32_t keysPerThread;
        uint32_t threadsPerThreadblock;
        uint32_t partitionSize;
        uint32_t totalSharedMemory;
    };

    typedef
        enum MODE
    {
        MODE_KEYS_ONLY = 0,
        MODE_PAIRS = 1,
    }   MODE;

    typedef
        enum ORDER
    {
        ORDER_ASCENDING = 0,
        ORDER_DESCENDING = 1,
    }   ORDER;

    typedef
        enum KEY_TYPE
    {
        KEY_UINT32 = 0,
        KEY_INT32 = 1,
        KEY_FLOAT32 = 2,
    }   KEY_TYPE;

    typedef
        enum PAYLOAD_TYPE
    {
        PAYLOAD_UINT32 = 0,
        PAYLOAD_INT32 = 1,
        PAYLOAD_FLOAT32 = 2,
    }   PAYLOAD_TYPE;

    struct GPUSortingConfig
    {
        MODE sortingMode;
        ORDER sortingOrder;
        KEY_TYPE sortingKeyType;
        PAYLOAD_TYPE sortingPayloadType;
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
}
