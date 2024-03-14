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
#include "ComputeShader.h"

namespace DeviceRadixSortKernels
{
    enum class Reg
    {
        Sort = 0,
        Alt = 1,
        SortPayload = 2,
        AltPayload = 3,
        GlobalHist = 4,
        PassHist = 5,
    };

    class InitDeviceRadixSort
    {
        ComputeShader* shader;
    public:
        InitDeviceRadixSort(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
            rootParameters[0].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/DeviceRadixSort.hlsl",
                L"InitDeviceRadixSort",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS globalHist,
            const uint32_t& threadBlocks)
        {
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRootUnorderedAccessView(0, globalHist);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };

    class Upsweep
    {
        ComputeShader* shader;
    public:
        explicit Upsweep(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::PassHist);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/DeviceRadixSort.hlsl",
                L"Upsweep",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS globalHist,
            D3D12_GPU_VIRTUAL_ADDRESS passHist,
            const uint32_t& numKeys,
            const uint32_t& threadBlocks,
            const uint32_t& radixShift)
        {

            std::array<uint32_t, 4> t = { numKeys, radixShift, threadBlocks, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
            cmdList->SetComputeRootUnorderedAccessView(3, passHist);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };

    class Scan
    {
        ComputeShader* shader;
    public:
        explicit Scan(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::PassHist);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/DeviceRadixSort.hlsl",
                L"Scan",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS passHist,
            const uint32_t& partitions)
        {
            std::array<uint32_t, 4> t = { 0, 0, partitions, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, passHist);
            cmdList->Dispatch(256, 1, 1);
        }
    };

    class Downsweep
    {
        ComputeShader* shader;
    public:
        explicit Downsweep(
            winrt::com_ptr<ID3D12Device> device,
            DeviceInfo const& info,
            std::vector<std::wstring> compileArguments)
        {
            auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(7);
            rootParameters[0].InitAsConstants(4, 0);
            rootParameters[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParameters[2].InitAsUnorderedAccessView((UINT)Reg::SortPayload);
            rootParameters[3].InitAsUnorderedAccessView((UINT)Reg::Alt);
            rootParameters[4].InitAsUnorderedAccessView((UINT)Reg::AltPayload);
            rootParameters[5].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            rootParameters[6].InitAsUnorderedAccessView((UINT)Reg::PassHist);

            shader = new ComputeShader(
                device,
                info,
                "Shaders/DeviceRadixSort.hlsl",
                L"Downsweep",
                compileArguments,
                rootParameters);
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS sortPayloadBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS altBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS altPayloadBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS globalHist,
            D3D12_GPU_VIRTUAL_ADDRESS passHist,
            const uint32_t& numKeys,
            const uint32_t& threadBlocks,
            const uint32_t& radixShift)
        {
            std::array<uint32_t, 4> t = { numKeys, radixShift, threadBlocks, 0 };
            shader->SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
            cmdList->SetComputeRootUnorderedAccessView(3, altBuffer);
            cmdList->SetComputeRootUnorderedAccessView(4, altPayloadBuffer);
            cmdList->SetComputeRootUnorderedAccessView(5, globalHist);
            cmdList->SetComputeRootUnorderedAccessView(6, passHist);
            cmdList->Dispatch(threadBlocks, 1, 1);
        }
    };
}