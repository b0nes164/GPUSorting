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
#include "OneSweep.h"


OneSweep::OneSweep(
    winrt::com_ptr<ID3D12Device> _device,
    DeviceInfo _deviceInfo,
    GPU_SORTING_ORDER sortingOrder,
    GPU_SORTING_KEY_TYPE keyType) :
    GPUSorter(
        _device,
        _deviceInfo,
        sortingOrder,
        keyType,
        "OneSweep",
        4,
        256,
        1 << 13)
{
    //TODO: better exception handling
    if (!m_devInfo.SupportsOneSweep)
        printf("Warning this device does not support OneSweep, correct execution is not guarunteed");

    m_device.copy_from(_device.get());
    SetCompileArguments();
    Initialize();
}

OneSweep::OneSweep(
    winrt::com_ptr<ID3D12Device> _device,
    DeviceInfo _deviceInfo,
    GPU_SORTING_ORDER sortingOrder,
    GPU_SORTING_KEY_TYPE keyType,
    GPU_SORTING_PAYLOAD_TYPE payloadType) :
    GPUSorter(
        _device,
        _deviceInfo,
        sortingOrder,
        keyType,
        payloadType,
        "OneSweep",
        4,
        256,
        1 << 13)
{
    //TODO: better exception handling
    if (!m_devInfo.SupportsOneSweep)
        printf("Warning this device does not support OneSweep, correct execution is not guarunteed");

    m_device.copy_from(_device.get());
    SetCompileArguments();
    Initialize();
}

OneSweep::~OneSweep()
{
}

bool OneSweep::TestAll()
{
    printf("Beginning ");
    printf(k_sortName);
    PrintSortingConfig(k_sortingConfig);
    printf("test all. \n");

    uint32_t sortPayloadTestsPassed = 0;
    const uint32_t testEnd = k_tuningParameters.partitionSize * 2 + 1;
    for (uint32_t i = k_tuningParameters.partitionSize; i < testEnd; ++i)
    {
        sortPayloadTestsPassed += ValidateSort(i, i);

        if (!(i & 127))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", sortPayloadTestsPassed, k_tuningParameters.partitionSize + 1);

    printf("Beginning large size tests\n");
    sortPayloadTestsPassed += ValidateSort(1 << 21, 5);

    sortPayloadTestsPassed += ValidateSort(1 << 22, 7);

    sortPayloadTestsPassed += ValidateSort(1 << 23, 11);

    uint32_t totalTests = k_tuningParameters.partitionSize + 1 + 3;
    if (sortPayloadTestsPassed == totalTests)
    {
        printf("%u / %u  All tests passed. \n\n", totalTests, totalTests);
        return true;
    }
    else
    {
        printf("%u / %u  Test failed. \n\n", sortPayloadTestsPassed, totalTests);
        return false;
    }
        
    return sortPayloadTestsPassed == totalTests;
}

void OneSweep::InitComputeShaders()
{
    m_initOneSweep = new OneSweepKernels::InitOneSweep(m_device, m_devInfo, m_compileArguments);
    m_globalHist = new OneSweepKernels::GlobalHist(m_device, m_devInfo, m_compileArguments);
    m_scan = new OneSweepKernels::Scan(m_device, m_devInfo, m_compileArguments);
    m_digitBinningPass = new OneSweepKernels::DigitBinningPass(m_device, m_devInfo, m_compileArguments);
    m_initSortInput = new InitSortInput(m_device, m_devInfo, m_compileArguments);
    m_clearErrorCount = new ClearErrorCount(m_device, m_devInfo, m_compileArguments);
    m_validate = new Validate(m_device, m_devInfo, m_compileArguments);
}

void OneSweep::UpdateSize(uint32_t size)
{
    if (m_numKeys != size)
    {
        m_numKeys = size;
        m_partitions = divRoundUp(m_numKeys, k_tuningParameters.partitionSize);
        m_globalHistPartitions = divRoundUp(m_numKeys, k_globalHistPartitionSize);
        DisposeBuffers();
        InitBuffers(m_numKeys, m_partitions);
    }
}

void OneSweep::DisposeBuffers()
{
    m_sortBuffer = nullptr;
    m_sortPayloadBuffer = nullptr;
    m_altBuffer = nullptr;
    m_altPayloadBuffer = nullptr;
    m_passHistBuffer = nullptr;
}

void OneSweep::InitStaticBuffers()
{
    m_globalHistBuffer = CreateBuffer(
        m_device,
        k_radix * k_radixPasses * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_indexBuffer = CreateBuffer(
        m_device,
        k_radixPasses * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_errorCountBuffer = CreateBuffer(
        m_device,
        1 * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_readBackBuffer = CreateBuffer(
        m_device,
        k_maxReadBack * sizeof(uint32_t),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE);
}

void OneSweep::InitBuffers(
    const uint32_t numKeys,
    const uint32_t threadBlocks)
{
    m_sortBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_altBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_passHistBuffer = CreateBuffer(
        m_device,
        k_radix * k_radixPasses * threadBlocks * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    if (k_sortingConfig.sortingMode == GPU_SORTING_PAIRS)
    {
        m_sortPayloadBuffer = CreateBuffer(
            m_device,
            numKeys * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        m_altPayloadBuffer = CreateBuffer(
            m_device,
            numKeys * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    else
    {
        m_sortPayloadBuffer = CreateBuffer(
            m_device,
            1 * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        m_altPayloadBuffer = CreateBuffer(
            m_device,
            1 * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
}

void OneSweep::PrepareSortCmdList()
{
    m_initOneSweep->Dispatch(
        m_cmdList,
        m_globalHistBuffer->GetGPUVirtualAddress(),
        m_passHistBuffer->GetGPUVirtualAddress(),
        m_indexBuffer->GetGPUVirtualAddress(),
        m_partitions);
    UAVBarrierSingle(m_cmdList, m_globalHistBuffer);

    m_globalHist->Dispatch(
        m_cmdList,
        m_sortBuffer->GetGPUVirtualAddress(),
        m_globalHistBuffer->GetGPUVirtualAddress(),
        m_numKeys,
        m_globalHistPartitions);
    UAVBarrierSingle(m_cmdList, m_globalHistBuffer);

    m_scan->Dispatch(
        m_cmdList,
        m_globalHistBuffer->GetGPUVirtualAddress(),
        m_passHistBuffer->GetGPUVirtualAddress(),
        m_partitions,
        k_radixPasses);
    UAVBarrierSingle(m_cmdList, m_passHistBuffer);

    for (uint32_t radixShift = 0; radixShift < 32; radixShift += 8)
    {
        m_digitBinningPass->Dispatch(
            m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_altBuffer->GetGPUVirtualAddress(),
            m_sortPayloadBuffer->GetGPUVirtualAddress(),
            m_altPayloadBuffer->GetGPUVirtualAddress(),
            m_passHistBuffer->GetGPUVirtualAddress(),
            m_indexBuffer->GetGPUVirtualAddress(),
            m_numKeys,
            m_partitions,
            radixShift);

        UAVBarrierSingle(m_cmdList, m_sortBuffer);
        UAVBarrierSingle(m_cmdList, m_sortPayloadBuffer);
        UAVBarrierSingle(m_cmdList, m_altBuffer);
        UAVBarrierSingle(m_cmdList, m_altPayloadBuffer);

        swap(m_sortBuffer, m_altBuffer);
        swap(m_sortPayloadBuffer, m_altPayloadBuffer);
    }
}