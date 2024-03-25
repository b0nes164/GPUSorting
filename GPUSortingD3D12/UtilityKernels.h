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

namespace UtilityKernels
{
    enum class Reg
    {
        Sort = 0,
        SortPayload = 1,
        PassHist = 2,
        ErrorCount = 3,
    };

    class InitSortInput : public ComputeKernelBase
    {
    public:

        explicit InitSortInput(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitSortInput",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortPayloadBuffer,
            const uint32_t& numKeys,
            const GPUSorting::ENTROPY_PRESET& entropyPreset,
            uint32_t seed)
        {
            std::array<uint32_t, 4> t = { numKeys, 0, seed, (uint32_t)entropyPreset };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::SortPayload);
            return rootParams;
        }
    };

    class ClearErrorCount : ComputeKernelBase
    {
    public:
        explicit ClearErrorCount(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ClearErrorCount",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount)
        {
            SetPipelineState(cmdList);
            cmdList->SetComputeRootUnorderedAccessView(0, errorCount);
            cmdList->Dispatch(1, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
            rootParams[0].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParams;
        }
    };

    class Validate : ComputeKernelBase
    {
        const uint32_t k_valPartSize = 2048;

    public:
        explicit Validate(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"Validate",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortPayloadBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& errorCount,
            const uint32_t& numKeys)
        {
            uint32_t valThreadBlocks = (numKeys + k_valPartSize - 1) / k_valPartSize;
            const uint32_t fullBlocks = valThreadBlocks / k_maxDim;
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = {
                    numKeys,
                    valThreadBlocks,
                    k_isNotPartialBitFlag,
                    0, };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);
            }
            
            const uint32_t partialBlocks = valThreadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                    numKeys,
                    valThreadBlocks,
                    fullBlocks << 1 | k_isPartialBitFlag,
                    0, };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::SortPayload);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::ErrorCount);
            return rootParams;
        }
    };

    class InitScanTestValues : public ComputeKernelBase
    {
    public:
        InitScanTestValues(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitScanTestValues",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& passHist,
            const uint32_t& numKeys)
        {
            std::array<uint32_t, 4> t = { numKeys, 0, 0, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, passHist);
            cmdList->Dispatch(1, 1, 1);
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
}