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
#include "GPUSorting.h"
#include "GPUSorter.h"
#include "DeviceRadixSortKernels.h"

class DeviceRadixSort : public GPUSorter
{
    winrt::com_ptr<ID3D12Resource> m_passHistBuffer;
    winrt::com_ptr<ID3D12Resource> m_globalHistBuffer;

    DeviceRadixSortKernels::InitDeviceRadixSort* m_initDeviceRadix;
    DeviceRadixSortKernels::Upsweep* m_upsweep;
    DeviceRadixSortKernels::Scan* m_scan;
    DeviceRadixSortKernels::Downsweep* m_downsweep;
    InitScanTestValues* m_initScanTestValues;

public:
    DeviceRadixSort(
        winrt::com_ptr<ID3D12Device> _device,
        DeviceInfo _deviceInfo,
        GPU_SORTING_ORDER sortingOrder,
        GPU_SORTING_KEY_TYPE keyType);

    DeviceRadixSort(
        winrt::com_ptr<ID3D12Device> _device,
        DeviceInfo _deviceInfo,
        GPU_SORTING_ORDER sortingOrder,
        GPU_SORTING_KEY_TYPE keyType,
        GPU_SORTING_PAYLOAD_TYPE payloadType);

    void TestAll() override;

protected:
    void InitComputeShaders() override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void InitBuffers(
        const uint32_t numKeys,
        const uint32_t threadBlocks) override;

    void PrepareSortCmdList() override;

    bool ValidateScan(uint32_t size);
};