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
#include "DeviceRadixSort.h"

DeviceRadixSort::DeviceRadixSort(
    winrt::com_ptr<ID3D12Device> _device, 
    DeviceInfo _deviceInfo, 
    GPU_SORTING_ORDER sortingOrder, 
    GPU_SORTING_KEY_TYPE keyType) :
    GPUSorter("DeviceRadixSort ", 4, 256, 3840, 1 << 13)
{
    m_device.copy_from(_device.get());
    m_devInfo = _deviceInfo;
    m_sortingConfig.sortingMode = GPU_SORTING_KEYS_ONLY;
    m_sortingConfig.sortingOrder = sortingOrder;
    m_sortingConfig.sortingKeyType = keyType;

    switch (keyType)
    {
    case GPU_SORTING_KEY_UINT32:
        m_compileArguments.push_back(L"-DKEY_UINT");
        break;
    case GPU_SORTING_KEY_INT32:
        m_compileArguments.push_back(L"-DKEY_INT");
        break;
    case GPU_SORTING_KEY_FLOAT32:
        m_compileArguments.push_back(L"-DKEY_FLOAT");
        break;
    }

    if (sortingOrder == GPU_SORTING_ASCENDING)
        m_compileArguments.push_back(L"-DSHOULD_ASCEND");

    Initialize();
}

DeviceRadixSort::DeviceRadixSort(
    winrt::com_ptr<ID3D12Device> _device,
    DeviceInfo _deviceInfo,
    GPU_SORTING_ORDER sortingOrder,
    GPU_SORTING_KEY_TYPE keyType,
    GPU_SORTING_PAYLOAD_TYPE payloadType) :
    GPUSorter("DeviceRadixSort ", 4, 256, 7680, 1 << 13)
{
    m_device.copy_from(_device.get());
    m_devInfo = _deviceInfo;
    m_sortingConfig.sortingMode = GPU_SORTING_PAIRS;
    m_sortingConfig.sortingOrder = sortingOrder;
    m_sortingConfig.sortingKeyType = keyType;
    m_sortingConfig.sortingPayloadType = payloadType;

    m_compileArguments.push_back(L"-DSORT_PAIRS");

    if (sortingOrder == GPU_SORTING_ASCENDING)
        m_compileArguments.push_back(L"-DSHOULD_ASCEND");

    switch (keyType)
    {
    case GPU_SORTING_KEY_UINT32:
        m_compileArguments.push_back(L"-DKEY_UINT");
        break;
    case GPU_SORTING_KEY_INT32:
        m_compileArguments.push_back(L"-DKEY_INT");
        break;
    case GPU_SORTING_KEY_FLOAT32:
        m_compileArguments.push_back(L"-DKEY_FLOAT");
        break;
    }

    switch (payloadType)
    {
    case GPU_SORTING_PAYLOAD_UINT32:
        m_compileArguments.push_back(L"-DPAYLOAD_UINT");
        break;
    case GPU_SORTING_PAYLOAD_INT32:
        m_compileArguments.push_back(L"-DPAYLOAD_INT");
        break;
    case GPU_SORTING_PAYLOAD_FLOAT32:
        m_compileArguments.push_back(L"-DPAYLOAD_FLOAT");
        break;
    }

    Initialize();
}

DeviceRadixSort::~DeviceRadixSort()
{
}

bool DeviceRadixSort::TestAll()
{
    printf("Beginning ");
    printf(k_sortName);
    PrintSortingConfig(m_sortingConfig);
    printf("test all. \n");

    uint32_t sortPayloadTestsPassed = 0;
    const uint32_t testEnd = k_partitionSize * 2 + 1;
    for (uint32_t i = k_partitionSize; i < testEnd; ++i)
    {
        sortPayloadTestsPassed += ValidateSort(i, i);

        if (!(i & 127))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", sortPayloadTestsPassed, k_partitionSize + 1);

    UpdateSize(1 << 22); //TODO: BAD!
    printf("Beginning interthreadblock scan validation tests. \n");
    uint32_t scanTestsPassed = 0;
    for (uint32_t i = 1; i < 256; ++i)
    {
        scanTestsPassed += ValidateScan(i);
        if (!(i & 7))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", scanTestsPassed, 255);

    printf("Beginning large size tests\n");
    sortPayloadTestsPassed += ValidateSort(1 << 21, 5);

    sortPayloadTestsPassed += ValidateSort(1 << 22, 7);

    sortPayloadTestsPassed += ValidateSort(1 << 23, 11);

    uint32_t totalTests = k_partitionSize + 1 + 255 + 3;
    if (sortPayloadTestsPassed + scanTestsPassed == totalTests)
    {
        printf("%u / %u  All tests passed. \n\n", totalTests, totalTests);
        return true;
    }
    else
    {
        printf("%u / %u  Test failed. \n\n", sortPayloadTestsPassed + scanTestsPassed, totalTests);
        return false;
    }
}

void DeviceRadixSort::InitComputeShaders()
{
    m_initDeviceRadix = new DeviceRadixSortKernels::InitDeviceRadixSort(m_device, m_devInfo, m_compileArguments);
    m_upsweep = new DeviceRadixSortKernels::Upsweep(m_device, m_devInfo, m_compileArguments);
    m_scan = new DeviceRadixSortKernels::Scan(m_device, m_devInfo, m_compileArguments);
    m_downsweep = new DeviceRadixSortKernels::Downsweep(m_device, m_devInfo, m_compileArguments);
    m_initSortInput = new InitSortInput(m_device, m_devInfo, m_compileArguments);
    m_initEntropy = new InitEntropyControlled(m_device, m_devInfo, m_compileArguments);
    m_clearErrorCount = new ClearErrorCount(m_device, m_devInfo, m_compileArguments);
    m_validate = new Validate(m_device, m_devInfo, m_compileArguments);
    m_initScanTestValues = new InitScanTestValues(m_device, m_devInfo, m_compileArguments);
}

void DeviceRadixSort::UpdateSize(uint32_t size)
{
    if (m_numKeys != size)
    {
        m_numKeys = size;
        m_partitions = divRoundUp(m_numKeys, k_partitionSize);
        DisposeBuffers();
        InitBuffers(m_numKeys, m_partitions);
    }
}

void DeviceRadixSort::DisposeBuffers()
{
    m_sortBuffer = nullptr;
    m_sortPayloadBuffer = nullptr;
    m_altBuffer = nullptr;
    m_altPayloadBuffer = nullptr;
    m_passHistBuffer = nullptr;
}

void DeviceRadixSort::InitStaticBuffers()
{
    m_globalHistBuffer = CreateBuffer(
        m_device,
        k_radix * k_radixPasses * sizeof(uint32_t),
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

void DeviceRadixSort::InitBuffers(const uint32_t numKeys, const uint32_t threadBlocks)
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
        k_radix * threadBlocks * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    if (m_sortingConfig.sortingMode == GPU_SORTING_PAIRS)
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

void DeviceRadixSort::PrepareSortCmdList()
{
    m_initDeviceRadix->Dispatch(
        m_cmdList,
        m_globalHistBuffer->GetGPUVirtualAddress(),
        1);
    UAVBarrierSingle(m_cmdList, m_globalHistBuffer);

    for (uint32_t radixShift = 0; radixShift < 32; radixShift += 8)
    {
        m_upsweep->Dispatch(
            m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_globalHistBuffer->GetGPUVirtualAddress(),
            m_passHistBuffer->GetGPUVirtualAddress(),
            m_numKeys,
            m_partitions,
            radixShift);
        UAVBarrierSingle(m_cmdList, m_passHistBuffer);

        m_scan->Dispatch(
            m_cmdList,
            m_passHistBuffer->GetGPUVirtualAddress(),
            m_partitions);
        UAVBarrierSingle(m_cmdList, m_passHistBuffer);
        UAVBarrierSingle(m_cmdList, m_globalHistBuffer);

        m_downsweep->Dispatch(
            m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_sortPayloadBuffer->GetGPUVirtualAddress(),
            m_altBuffer->GetGPUVirtualAddress(),
            m_altPayloadBuffer->GetGPUVirtualAddress(),
            m_globalHistBuffer->GetGPUVirtualAddress(),
            m_passHistBuffer->GetGPUVirtualAddress(),
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

bool DeviceRadixSort::ValidateScan(uint32_t size)
{
    m_initScanTestValues->Dispatch(
        m_cmdList,
        m_passHistBuffer->GetGPUVirtualAddress(),
        size);
    UAVBarrierSingle(m_cmdList, m_passHistBuffer);

    m_scan->Dispatch(
        m_cmdList,
        m_passHistBuffer->GetGPUVirtualAddress(),
        size);
    ExecuteCommandList();

    m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_passHistBuffer.get(), 0, (uint64_t)size * sizeof(uint32_t));
    ExecuteCommandList();

    std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, size);

    bool isValid = true;
    for (uint32_t i = 0; i < size; ++i)
    {
        if (vecOut[i] != i)
        {
            printf("\nFailed at size %u.\n", size);
            isValid = false;

            break;
        }
    }

    return isValid;
}