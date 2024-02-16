#pragma once
#include "pch.h"

static winrt::com_ptr<ID3D12Resource> CreateBuffer(
    winrt::com_ptr<ID3D12Device> device, 
    uint32_t size, 
    D3D12_HEAP_TYPE heapType, 
    D3D12_RESOURCE_STATES initialState, 
    D3D12_RESOURCE_FLAGS flags)
{
    winrt::com_ptr<ID3D12Resource> buffer;
    auto bufferHeapProps = CD3DX12_HEAP_PROPERTIES(heapType);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(size, flags);
    winrt::check_hresult(device->CreateCommittedResource(
        &bufferHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        initialState,
        nullptr,
        IID_PPV_ARGS(buffer.put())));
    return buffer;
}

static void UAVBarrierSingle(
    winrt::com_ptr<ID3D12GraphicsCommandList> commandList,
    winrt::com_ptr<ID3D12Resource> resource)
{
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource.get();
    commandList->ResourceBarrier(1, &barrier);
}

static std::vector<uint32_t> ReadBackBuffer(
    winrt::com_ptr<ID3D12Resource> readBackBuffer,
    uint32_t readBackSize)
{
    std::vector<uint32_t> result(readBackSize);
    void* pData;
    winrt::check_hresult(readBackBuffer->Map(0, nullptr, &pData));
    memcpy(result.data(), pData, readBackSize * sizeof(uint32_t));
    readBackBuffer->Unmap(0, nullptr);
    return result;
}

static std::vector<uint64_t> ReadBackTiming(winrt::com_ptr<ID3D12Resource> readBackBuffer)
{
    std::vector<uint64_t> result(2);
    void* pData;
    winrt::check_hresult(readBackBuffer->Map(0, nullptr, &pData));
    memcpy(result.data(), pData, 2 * sizeof(uint64_t));
    readBackBuffer->Unmap(0, nullptr);
    return result;
}