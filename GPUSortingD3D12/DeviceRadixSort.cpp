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
    GPU_SORTING_KEY_TYPE keyType)
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

    if(sortingOrder == GPU_SORTING_ASCENDING)
        m_compileArguments.push_back(L"-DSHOULD_ASCEND");

    Initialize();
}

DeviceRadixSort::DeviceRadixSort(
    winrt::com_ptr<ID3D12Device> _device, 
    DeviceInfo _deviceInfo,
    GPU_SORTING_ORDER sortingOrder,
    GPU_SORTING_KEY_TYPE keyType, 
    GPU_SORTING_PAYLOAD_TYPE payloadType)
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

void DeviceRadixSort::TestSort(uint32_t testSize, uint32_t seed, bool shouldReadBack, bool shouldValidate)
{
    UpdateSize(testSize);
    CreateTestInput(seed);
    PrepareSortCmdList();
    ExecuteCommandList();

    if (shouldValidate)
        ValidateOutput(true);

    if (shouldReadBack)
    {
        uint64_t readBackSize = numKeys < maxReadback ? numKeys : maxReadback;
        m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_sortBuffer.get(), 0, readBackSize * sizeof(uint32_t));
        ExecuteCommandList();
        std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, numKeys);

        printf("---------------KEYS---------------\n");
        for (uint32_t i = 0; i < vecOut.size(); ++i)
            printf("%u %u \n", i, vecOut[i]);

        if (m_sortingConfig.sortingMode == GPU_SORTING_PAIRS)
        {
            m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_sortPayloadBuffer.get(), 0, readBackSize * sizeof(uint32_t));
            ExecuteCommandList();
            vecOut = ReadBackBuffer(m_readBackBuffer, numKeys);

            printf("\n \n \n");
            printf("---------------PAYLOADS---------------\n");
            for (uint32_t i = 0; i < vecOut.size(); ++i)
                printf("%u %u \n", i, vecOut[i]);
        }
    }
}

//TODO: move to abstract
void DeviceRadixSort::BatchTiming(uint32_t inputSize, uint32_t batchSize)
{
    UpdateSize(inputSize);

    printf("Beginning timing test \n");
    double totalTime = 0.0;
    for (uint32_t i = 0; i <= batchSize; ++i)
    {
        double t = TimeSort(i + 10);
        if (i)
            totalTime += t;

        if ((i & 7) == 0)
            printf(".");
    }
    printf("\n");

    totalTime = inputSize / totalTime * batchSize;
    printf("Estimated speed at %u iterations and %u keys: %E \n", batchSize, inputSize, totalTime);
}

void DeviceRadixSort::TestAll()
{
    printf("Beginning ");
    printf(k_sortName);
    PrintSortingConfig(m_sortingConfig);
    printf("test all. \n");

    uint32_t sortPayloadTestsPassed = 0;
    const uint32_t testEnd = partitionSize * 2 + 1;
    for (uint32_t i = partitionSize; i < testEnd; ++i)
    {
        sortPayloadTestsPassed += ValidateSort(i, i);

        if (!(i & 127))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", sortPayloadTestsPassed, partitionSize + 1);

    UpdateSize(1 << 21); //TODO: BAD!
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

    UpdateSize(1);
    UpdateSize(1 << 22);
    sortPayloadTestsPassed += ValidateSort(1 << 22, 7);

    UpdateSize(1);
    UpdateSize(1 << 23);
    sortPayloadTestsPassed += ValidateSort(1 << 23, 11);

    uint32_t totalTests = partitionSize + 1 + 255 + 3;
    if (sortPayloadTestsPassed + scanTestsPassed == totalTests)
        printf("%u / %u  All tests passed. \n", totalTests, totalTests);
    else
        printf("%u / %u  Test failed. \n", sortPayloadTestsPassed + scanTestsPassed, totalTests);
}

void DeviceRadixSort::Initialize()
{
    m_initDeviceRadix = new InitDeviceRadixSort(m_device, m_devInfo, m_compileArguments);
    m_initSortInput = new InitSortInput(m_device, m_devInfo, m_compileArguments);
    m_upsweep = new Upsweep(m_device, m_devInfo, m_compileArguments);
    m_scan = new Scan(m_device, m_devInfo, m_compileArguments);
    m_downsweep = new Downsweep(m_device, m_devInfo, m_compileArguments);
    m_clearErrorCount = new ClearErrorCount(m_device, m_devInfo, m_compileArguments);
    m_validate = new Validate(m_device, m_devInfo, m_compileArguments);
    m_initScanTestValues = new InitScanTestValues(m_device, m_devInfo, m_compileArguments);

    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    winrt::check_hresult(m_device->CreateCommandQueue(&desc, IID_PPV_ARGS(m_cmdQueue.put())));
    winrt::check_hresult(m_device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(m_cmdAllocator.put())));
    winrt::check_hresult(m_device->CreateCommandList(0, desc.Type, m_cmdAllocator.get(), nullptr, IID_PPV_ARGS(m_cmdList.put())));
    winrt::check_hresult(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.put())));
    m_fenceEvent.reset(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    m_nextFenceValue = 1;

    D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
    queryHeapDesc.Count = 2;
    queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    winrt::check_hresult(m_device->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(m_queryHeap.put())));
    winrt::check_hresult(m_cmdQueue->GetTimestampFrequency(&m_timestampFrequency));

    InitStaticBuffers();
}

//TODO: move to abstract
void DeviceRadixSort::UpdateSize(uint32_t size)
{
    if (numKeys != size)
    {
        numKeys = size;
        partitions = (numKeys + partitionSize - 1) / partitionSize;
        DisposeBuffers();
        InitBuffers(numKeys, partitions);
    }
}

void DeviceRadixSort::DisposeBuffers()
{
    m_sortBuffer = nullptr;
    m_altBuffer = nullptr;
    m_passHistBuffer = nullptr;
}

void DeviceRadixSort::InitStaticBuffers()
{
    m_globalHistBuffer = CreateBuffer(
        m_device,
        radix * radixPasses * sizeof(uint32_t),
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
        maxReadback * sizeof(uint32_t),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE);
}

void DeviceRadixSort::InitBuffers(uint32_t numKeys, const uint32_t threadBlocks)
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
        radix * threadBlocks * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    //TODO : BAD!
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

//TODO: move to abstract
//Initialize the input for testing
void DeviceRadixSort::CreateTestInput(uint32_t seed)
{
    //Init the sorting input
    m_initSortInput->Dispatch(m_cmdList,
        m_sortBuffer->GetGPUVirtualAddress(),
        m_sortPayloadBuffer->GetGPUVirtualAddress(),
        numKeys,
        seed);
    UAVBarrierSingle(m_cmdList, m_sortBuffer);
    ExecuteCommandList();
}

//Adds all necessary items to the cmdList, but does not execute.
void DeviceRadixSort::PrepareSortCmdList()
{
    m_initDeviceRadix->Dispatch(m_cmdList,
        m_globalHistBuffer->GetGPUVirtualAddress(),
        1);
    UAVBarrierSingle(m_cmdList, m_globalHistBuffer);

    for (uint32_t radixShift = 0; radixShift < 32; radixShift += 8)
    {
        m_upsweep->Dispatch(m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_globalHistBuffer->GetGPUVirtualAddress(),
            m_passHistBuffer->GetGPUVirtualAddress(),
            numKeys,
            partitions,
            radixShift);
        UAVBarrierSingle(m_cmdList, m_passHistBuffer);

        m_scan->Dispatch(m_cmdList,
            m_passHistBuffer->GetGPUVirtualAddress(),
            partitions);
        UAVBarrierSingle(m_cmdList, m_passHistBuffer);
        UAVBarrierSingle(m_cmdList, m_globalHistBuffer);

        m_downsweep->Dispatch(m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_sortPayloadBuffer->GetGPUVirtualAddress(),
            m_altBuffer->GetGPUVirtualAddress(),
            m_altPayloadBuffer->GetGPUVirtualAddress(),
            m_globalHistBuffer->GetGPUVirtualAddress(),
            m_passHistBuffer->GetGPUVirtualAddress(),
            numKeys,
            partitions,
            radixShift);
        UAVBarrierSingle(m_cmdList, m_sortBuffer);
        UAVBarrierSingle(m_cmdList, m_sortPayloadBuffer);
        UAVBarrierSingle(m_cmdList, m_altBuffer);
        UAVBarrierSingle(m_cmdList, m_altPayloadBuffer);

        swap(m_sortBuffer, m_altBuffer);
        swap(m_sortPayloadBuffer, m_altPayloadBuffer);
    }
}

//TODO: move to abstract
void DeviceRadixSort::ExecuteCommandList()
{
    winrt::check_hresult(m_cmdList->Close());
    ID3D12CommandList* commandLists[] = { m_cmdList.get() };
    m_cmdQueue->ExecuteCommandLists(1, commandLists);
    winrt::check_hresult(m_cmdQueue->Signal(m_fence.get(), m_nextFenceValue));
    winrt::check_hresult(m_fence->SetEventOnCompletion(m_nextFenceValue, m_fenceEvent.get()));
    ++m_nextFenceValue;
    winrt::check_hresult(m_fenceEvent.wait());
    winrt::check_hresult(m_cmdAllocator->Reset());
    winrt::check_hresult(m_cmdList->Reset(m_cmdAllocator.get(), nullptr));
}

//TODO: move to abstract
bool DeviceRadixSort::ValidateOutput(bool shouldPrint)
{
    m_clearErrorCount->Dispatch(m_cmdList,
        m_errorCountBuffer->GetGPUVirtualAddress());
    UAVBarrierSingle(m_cmdList, m_errorCountBuffer);

    m_validate->Dispatch(m_cmdList,
        m_sortBuffer->GetGPUVirtualAddress(),
        m_sortPayloadBuffer->GetGPUVirtualAddress(),
        m_errorCountBuffer->GetGPUVirtualAddress(),
        numKeys);

    UAVBarrierSingle(m_cmdList, m_errorCountBuffer);
    ExecuteCommandList();

    m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_errorCountBuffer.get(), 0, sizeof(uint32_t));
    ExecuteCommandList();
    std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, 1);
    uint32_t errCount = vecOut[0];

    if (shouldPrint)
    {
        printf(k_sortName);
        PrintSortingConfig(m_sortingConfig);
        if (errCount)
            printf("failed at size %u with %u errors. \n", numKeys, errCount);
        else
            printf("passed at size %u. \n", numKeys);
    }

    return !errCount;
}

//TODO: move to abstract
bool DeviceRadixSort::ValidateSort(uint32_t size, uint32_t seed)
{
    UpdateSize(size);
    CreateTestInput(seed);
    PrepareSortCmdList();
    ExecuteCommandList();
    return ValidateOutput(false);
}

bool DeviceRadixSort::ValidateScan(uint32_t size)
{
    m_initScanTestValues->Dispatch(m_cmdList,
        m_passHistBuffer->GetGPUVirtualAddress(),
        size);
    UAVBarrierSingle(m_cmdList, m_passHistBuffer);

    m_scan->Dispatch(m_cmdList,
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

//TODO: move to abstract
double DeviceRadixSort::TimeSort(uint32_t seed)
{
    CreateTestInput(seed);
    m_cmdList->EndQuery(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    PrepareSortCmdList();
    m_cmdList->EndQuery(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    ExecuteCommandList();

    m_cmdList->ResolveQueryData(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, m_readBackBuffer.get(), 0);
    ExecuteCommandList();

    std::vector<uint64_t> vecOut = ReadBackTiming(m_readBackBuffer);
    uint64_t diff = vecOut[1] - vecOut[0];
    return diff / (double)m_timestampFrequency;
}
