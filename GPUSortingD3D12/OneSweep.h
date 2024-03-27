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
#include "OneSweepKernels.h"

class OneSweep : public GPUSortBase
{
    const uint32_t k_globalHistPartitionSize = 32768;

    winrt::com_ptr<ID3D12Resource> m_indexBuffer;
    winrt::com_ptr<ID3D12Resource> m_passHistBuffer;
    winrt::com_ptr<ID3D12Resource> m_globalHistBuffer;

    OneSweepKernels::InitOneSweep* m_initOneSweep;
    OneSweepKernels::GlobalHist* m_globalHist;
    OneSweepKernels::Scan* m_scan;
    OneSweepKernels::DigitBinningPass* m_digitBinningPass;

    uint32_t m_globalHistPartitions;

public:
    OneSweep(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType);

    OneSweep(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType,
        GPUSorting::PAYLOAD_TYPE payloadType);

    ~OneSweep();

    bool TestAll() override;

protected:
    void InitComputeShaders() override;

    void UpdateSize(uint32_t size) override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void InitBuffers(
        const uint32_t numKeys,
        const uint32_t threadBlocks) override;

    void PrepareSortCmdList() override;
};