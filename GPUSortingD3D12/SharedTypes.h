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
    bool SupportsGPUSort;
};