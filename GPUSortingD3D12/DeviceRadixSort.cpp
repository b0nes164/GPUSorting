#pragma once
#include "pch.h"
#include "DeviceRadixSort.h"

DeviceRadixSort::DeviceRadixSort(winrt::com_ptr<ID3D12Device> _device, DeviceInfo _deviceInfo)
{
    m_device.copy_from(_device.get());
    m_devInfo = _deviceInfo;
    
    //TODO better integrate compiler args
    std::vector<std::wstring> compileArguments;
    //compileArguments.push_back(L"-DSORT_PAIRS");
    compileArguments.push_back(L"-DKEY_UINT");
    //compileArguments.push_back(L"-DPAYLOAD_UINT");
    compileArguments.push_back(L"-DSHOULD_ASCEND");

    m_initDeviceRadix = new InitDeviceRadixSort(m_device, m_devInfo, compileArguments);
    m_initSortInput = new InitSortInput(m_device, m_devInfo, compileArguments);
    m_upsweep = new Upsweep(m_device, m_devInfo, compileArguments);
    m_scan = new Scan(m_device, m_devInfo, compileArguments);
    m_downsweep = new Downsweep(m_device, m_devInfo, compileArguments);
    m_clearErrorCount = new ClearErrorCount(m_device, m_devInfo, compileArguments);
    m_validate = new Validate(m_device, m_devInfo, compileArguments);
    m_initScanTestValues = new InitScanTestValues(m_device, m_devInfo, compileArguments);

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
}

void DeviceRadixSort::TestSort(uint32_t testSize, uint32_t seed, bool shouldReadBack, bool shouldValidate)
{
    UpdateSize(1); //TODO: BAD!
    UpdateSize(testSize);
    CreateTestInput(seed);
    PrepareSortCmdList();
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    if (shouldValidate)
        ValidateOutput(m_sortBuffer, true, "Keys");

    if (shouldReadBack)
    {
        m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_sortBuffer.get(), 0, (uint64_t)numKeys * sizeof(uint32_t));
        ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

        std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, numKeys);

        for (uint32_t i = 0; i < vecOut.size(); ++i)
            printf("%u %u \n", i, vecOut[i]);
    }
}

void DeviceRadixSort::TestSortPayload(uint32_t testSize, uint32_t seed, bool shouldReadBack, bool shouldValidate)
{
    UpdateSize(1); //TODO: BAD!
    UpdateSize(testSize);
    CreateTestInput(seed);
    PrepareSortCmdList();
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    if (shouldValidate)
    {
        ValidateOutput(m_sortBuffer, true, "Keys");
        ValidateOutput(m_sortPayloadBuffer, true, "Payload");
    }
        
    if (shouldReadBack)
    {
        m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_sortBuffer.get(), 0, (uint64_t)numKeys * sizeof(uint32_t));
        ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

        std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, numKeys);

        for (uint32_t i = 0; i < vecOut.size(); ++i)
            printf("%u %u \n", i, vecOut[i]);
    }
}

void DeviceRadixSort::BatchTiming(uint32_t inputSize, uint32_t batchSize)
{
    UpdateSize(1); //TODO: BAD!
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

double DeviceRadixSort::TimeSort(uint32_t seed)
{
    CreateTestInput(seed);
    m_cmdList->EndQuery(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    PrepareSortCmdList();
    m_cmdList->EndQuery(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    m_cmdList->ResolveQueryData(m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, m_readBackBuffer.get(), 0);
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    std::vector<uint64_t> vecOut = ReadBackTiming(m_readBackBuffer);
    uint64_t diff = vecOut[1] - vecOut[0];
    return diff / (double)m_timestampFrequency;
}

void DeviceRadixSort::TestAll()
{
    printf("Beggining sort and payload validation tests. \n");
    uint32_t sortPayloadTestsPassed = 0;
    const uint32_t testEnd = partitionSize * 2 + 1;
    for (uint32_t i = partitionSize; i < testEnd; ++i)
    {
        UpdateSize(i);
        sortPayloadTestsPassed += ValidateSortAndPayload(i);

        if (!(i & 127))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", sortPayloadTestsPassed, partitionSize + 1);

    UpdateSize(1 << 21); //TODO: BAD!
    printf("Beggining interthreadblock scan validation tests. \n");
    uint32_t scanTestsPassed = 0;
    for (uint32_t i = 1; i < 256; ++i)
    {
        scanTestsPassed += ValidateScan(i);
        if (!(i & 7))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", scanTestsPassed, 255);

    printf("Beggining large size tests\n");
    sortPayloadTestsPassed += ValidateSortAndPayload(5);

    UpdateSize(1);
    UpdateSize(1 << 22);
    sortPayloadTestsPassed += ValidateSortAndPayload(7);

    UpdateSize(1);
    UpdateSize(1 << 23);
    sortPayloadTestsPassed += ValidateSortAndPayload(11);

    uint32_t totalTests = partitionSize + 1 + 255 + 3;
    if (sortPayloadTestsPassed + scanTestsPassed == totalTests)
        printf("%u / %u  All tests passed. \n", totalTests, totalTests);
    else
        printf("%u / %u  Test failed. \n", sortPayloadTestsPassed + scanTestsPassed, totalTests);
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
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_passHistBuffer.get(), 0, size * sizeof(uint32_t));
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, size);

    bool isValid = true;
    for (uint32_t i = 0; i < size; ++i)
    {
        if (vecOut[i] != i)
        {
            isValid = false;
            break;
        }
    }

    return isValid;
}

bool DeviceRadixSort::ValidateOutput(winrt::com_ptr<ID3D12Resource> toValidate, bool shouldPrint, const char* whatValidated)
{
    m_clearErrorCount->Dispatch(m_cmdList,
        m_errorCountBuffer->GetGPUVirtualAddress());
    UAVBarrierSingle(m_cmdList, m_errorCountBuffer);

    m_validate->Dispatch(m_cmdList,
        toValidate->GetGPUVirtualAddress(),
        m_errorCountBuffer->GetGPUVirtualAddress(),
        m_errorBuffer->GetGPUVirtualAddress(),
        numKeys,
        maxErrorReadback);
    UAVBarrierSingle(m_cmdList, m_errorCountBuffer);
    UAVBarrierSingle(m_cmdList, m_errorBuffer);
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_errorCountBuffer.get(), 0, sizeof(uint32_t));
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);
    std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, 1);
    uint32_t errCount = vecOut[0];

    if (shouldPrint)
    {
        if (errCount)
        {
            printf(whatValidated);
            printf(" failed: %u errors counted! \n", errCount);
            errCount = (errCount > maxErrorReadback ? maxErrorReadback : errCount);
            m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_errorBuffer.get(), 0, errCount * 3 * sizeof(uint32_t));
            ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

            std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, errCount * 3);
            for (uint32_t i = 0; i < errCount; ++i)
                printf("Error at index: %u. Value: %u Next Value: %u \n", vecOut[i * 3], vecOut[i * 3 + 1], vecOut[i * 3 + 2]);
        }
        else
        {
            printf(whatValidated);
            printf(" passed at size %u \n", numKeys);
        }
    }

    return !errCount;
}

bool DeviceRadixSort::ValidateSortAndPayload(uint32_t seed)
{
    CreateTestInput(seed);
    PrepareSortCmdList();
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);

    return ValidateOutput(m_sortBuffer, false, "") &&
        ValidateOutput(m_sortPayloadBuffer, false, "");
}

void DeviceRadixSort::UpdateSize(uint32_t size)
{
    if (numKeys != size)
    {
        numKeys = size;
        partitions = (numKeys + partitionSize - 1) / partitionSize;
        InitBuffers(numKeys, radixPasses, radix, partitions);
    }
}

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
    ExecuteCommandListSynchronously(m_cmdList, m_cmdQueue, m_cmdAllocator, m_fence, m_fenceEvent, m_nextFenceValue);
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

void DeviceRadixSort::InitBuffers(
    const uint32_t& numKeys, 
    const uint32_t& radixPasses, 
    const uint32_t radixDigits, 
    const uint32_t threadBlocks)
{
    m_sortBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_sortPayloadBuffer = CreateBuffer(
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

    m_altPayloadBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_passHistBuffer = CreateBuffer(
        m_device,
        radixDigits * threadBlocks * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_globalHistBuffer = CreateBuffer(
        m_device,
        radixDigits * radixPasses * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_errorCountBuffer = CreateBuffer(
        m_device,
        1 * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_errorBuffer = CreateBuffer(
        m_device,
        maxErrorReadback * 3 * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_readBackBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE);
}