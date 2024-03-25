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
#include "GPUSortBase.h"
#include "DeviceRadixSortKernels.h"

class DeviceRadixSort : public GPUSortBase
{
    winrt::com_ptr<ID3D12Resource> m_passHistBuffer;
    winrt::com_ptr<ID3D12Resource> m_globalHistBuffer;

    DeviceRadixSortKernels::InitDeviceRadixSort* m_initDeviceRadix;
    DeviceRadixSortKernels::Upsweep* m_upsweep;
    DeviceRadixSortKernels::Scan* m_scan;
    DeviceRadixSortKernels::Downsweep* m_downsweep;
    UtilityKernels::InitScanTestValues* m_initScanTestValues;

public:
    DeviceRadixSort(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType);

    DeviceRadixSort(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType,
        GPUSorting::PAYLOAD_TYPE payloadType);

    ~DeviceRadixSort();

    bool TestAll() override;

protected:
    void InitUtilityComputeShaders() override;

    void InitComputeShaders() override;

    void UpdateSize(uint32_t size) override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void InitBuffers(
        const uint32_t numKeys,
        const uint32_t threadBlocks) override;

    void PrepareSortCmdList() override;

    bool ValidateScan(uint32_t size);
};