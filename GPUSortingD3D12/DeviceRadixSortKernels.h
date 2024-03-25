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
#include "ComputeKernelBase.h"

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

    class InitDeviceRadixSort : public ComputeKernelBase
    {
    public:
        InitDeviceRadixSort(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring> compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitDeviceRadixSort",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& globalHist)
        {
            SetPipelineState(cmdList);
            cmdList->SetComputeRootUnorderedAccessView(0, globalHist);
            cmdList->Dispatch(1, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
            rootParams[0].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            return rootParams;
        }
    };

    class Upsweep : public ComputeKernelBase
    {
    public:
        Upsweep(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring> compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"Upsweep",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& globalHist,
            const D3D12_GPU_VIRTUAL_ADDRESS& passHist,
            const uint32_t& numKeys,
            const uint32_t& threadBlocks,
            const uint32_t& radixShift)
        {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = { 
                    numKeys,
                    radixShift,
                    threadBlocks,
                    k_isNotPartialBitFlag };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
                cmdList->SetComputeRootUnorderedAccessView(3, passHist);
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                numKeys,
                radixShift,
                threadBlocks,
                fullBlocks << 1 | k_isPartialBitFlag };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
                cmdList->SetComputeRootUnorderedAccessView(3, passHist);
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::PassHist);
            return rootParams;
        }
    };

    class Scan : public ComputeKernelBase
    {
    public:
        Scan(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"Scan",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& passHist,
            const uint32_t& partitions)
        {
            std::array<uint32_t, 4> t = { 0, 0, partitions, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, passHist);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::PassHist);
            return rootParams;
        }
    };

    class Downsweep : ComputeKernelBase
    {
    public:
        Downsweep(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"Downsweep",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortPayloadBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& altBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& altPayloadBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& globalHist,
            const D3D12_GPU_VIRTUAL_ADDRESS& passHist,
            const uint32_t& numKeys,
            const uint32_t& threadBlocks,
            const uint32_t& radixShift)
        {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = {
                    numKeys,
                    radixShift,
                    threadBlocks,
                    k_isNotPartialBitFlag };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, altBuffer);
                cmdList->SetComputeRootUnorderedAccessView(4, altPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(5, globalHist);
                cmdList->SetComputeRootUnorderedAccessView(6, passHist);
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                numKeys,
                radixShift,
                threadBlocks,
                fullBlocks << 1 | k_isPartialBitFlag };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, altBuffer);
                cmdList->SetComputeRootUnorderedAccessView(4, altPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(5, globalHist);
                cmdList->SetComputeRootUnorderedAccessView(6, passHist);
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(7);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::SortPayload);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::Alt);
            rootParams[4].InitAsUnorderedAccessView((UINT)Reg::AltPayload);
            rootParams[5].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            rootParams[6].InitAsUnorderedAccessView((UINT)Reg::PassHist);
            return rootParams;
        }
    };
}