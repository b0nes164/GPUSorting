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
#include "Utils.h"
#include "UtilityKernels.h"

class GPUSorter
{
protected:
    const char* k_sortName;
    const uint32_t k_radixPasses;
    const uint32_t k_radix;
    const uint32_t k_partitionSize;
    const uint32_t k_maxReadBack;

    uint32_t m_numKeys = 0;
    uint32_t m_partitions = 0;

    winrt::com_ptr<ID3D12Device> m_device;
    DeviceInfo m_devInfo{};
    std::vector<std::wstring> m_compileArguments;
    GPUSortingConfig m_sortingConfig{};

    winrt::com_ptr<ID3D12GraphicsCommandList> m_cmdList;
    winrt::com_ptr<ID3D12CommandQueue> m_cmdQueue;
    winrt::com_ptr<ID3D12CommandAllocator> m_cmdAllocator;

    winrt::com_ptr<ID3D12QueryHeap> m_queryHeap;
    winrt::com_ptr<ID3D12Fence> m_fence;
    wil::unique_event_nothrow m_fenceEvent;
    uint64_t m_nextFenceValue;
    uint64_t m_timestampFrequency;

    winrt::com_ptr<ID3D12Resource> m_sortBuffer;
    winrt::com_ptr<ID3D12Resource> m_sortPayloadBuffer;
    winrt::com_ptr<ID3D12Resource> m_altBuffer;
    winrt::com_ptr<ID3D12Resource> m_altPayloadBuffer;
    winrt::com_ptr<ID3D12Resource> m_errorCountBuffer;
    winrt::com_ptr<ID3D12Resource> m_readBackBuffer;

    InitSortInput* m_initSortInput;
    ClearErrorCount* m_clearErrorCount;
    Validate* m_validate;

    GPUSorter(
        const char* sortName,
        uint32_t radixPasses,
        uint32_t radix,
        uint32_t partitionSize,
        uint32_t maxReadBack) :
        k_sortName(sortName),
        k_radixPasses(radixPasses),
        k_radix(radix),
        k_partitionSize(partitionSize),
        k_maxReadBack(maxReadBack)
    {
    };

public:
    void TestSort(
        uint32_t testSize,
        uint32_t seed,
        bool shouldReadBack,
        bool shouldValidate)
    {
        UpdateSize(testSize);
        CreateTestInput(seed);
        PrepareSortCmdList();
        ExecuteCommandList();

        if (shouldValidate)
            ValidateOutput(true);

        if (shouldReadBack)
        {
            uint64_t readBackSize = m_numKeys < k_maxReadBack ? m_numKeys : k_maxReadBack;
            ReadbackPreBarrier(m_cmdList, m_sortBuffer);
            m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_sortBuffer.get(), 0, readBackSize * sizeof(uint32_t));
            ReadbackPostBarrier(m_cmdList, m_sortBuffer);
            ExecuteCommandList();
            std::vector<uint32_t> vecOut = ReadBackBuffer(m_readBackBuffer, readBackSize);

            printf("---------------KEYS---------------\n");
            for (uint32_t i = 0; i < vecOut.size(); ++i)
                printf("%u %u \n", i, vecOut[i]);

            if (m_sortingConfig.sortingMode == GPU_SORTING_PAIRS)
            {
                ReadbackPreBarrier(m_cmdList, m_sortPayloadBuffer);
                m_cmdList->CopyBufferRegion(m_readBackBuffer.get(), 0, m_sortPayloadBuffer.get(), 0, readBackSize * sizeof(uint32_t));
                ReadbackPostBarrier(m_cmdList, m_sortPayloadBuffer);
                ExecuteCommandList();
                vecOut = ReadBackBuffer(m_readBackBuffer, readBackSize);

                printf("\n \n \n");
                printf("---------------PAYLOADS---------------\n");
                for (uint32_t i = 0; i < vecOut.size(); ++i)
                    printf("%u %u \n", i, vecOut[i]);
            }
        }
    }

    void BatchTiming(uint32_t inputSize, uint32_t batchSize)
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

    virtual void TestAll() = 0;

protected:
    void Initialize()
    {
        InitComputeShaders();

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

    virtual void InitComputeShaders() = 0;

    void UpdateSize(uint32_t size)
    {
        if (m_numKeys != size)
        {
            m_numKeys = size;
            m_partitions = (m_numKeys + k_partitionSize - 1) / k_partitionSize;
            DisposeBuffers();
            InitBuffers(m_numKeys, m_partitions);
        }
    }

    virtual void DisposeBuffers() = 0;

    virtual void InitStaticBuffers() = 0;

    virtual void InitBuffers(
        const uint32_t numKeys,
        const uint32_t threadBlocks) = 0;

    void CreateTestInput(uint32_t seed)
    {
        //Init the sorting input
        m_initSortInput->Dispatch(m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_sortPayloadBuffer->GetGPUVirtualAddress(),
            m_numKeys,
            seed);
        UAVBarrierSingle(m_cmdList, m_sortBuffer);
        ExecuteCommandList();
    }

    virtual void PrepareSortCmdList() = 0;

    void ExecuteCommandList()
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

    bool ValidateOutput(bool shouldPrint)
    {
        m_clearErrorCount->Dispatch(m_cmdList,
            m_errorCountBuffer->GetGPUVirtualAddress());
        UAVBarrierSingle(m_cmdList, m_errorCountBuffer);

        m_validate->Dispatch(m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_sortPayloadBuffer->GetGPUVirtualAddress(),
            m_errorCountBuffer->GetGPUVirtualAddress(),
            m_numKeys);

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
                printf("failed at size %u with %u errors. \n", m_numKeys, errCount);
            else
                printf("passed at size %u. \n", m_numKeys);
        }

        return !errCount;
    }

    bool ValidateSort(uint32_t size, uint32_t seed)
    {
        UpdateSize(size);
        CreateTestInput(seed);
        PrepareSortCmdList();
        ExecuteCommandList();
        return ValidateOutput(false);
    }

    double TimeSort(uint32_t seed)
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
};