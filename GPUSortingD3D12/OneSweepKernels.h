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
#include "Utils.h"

namespace OneSweepKernels
{
    enum class Reg
    {
        Sort = 0,
        Alt = 1,
        SortPayload = 2,
        AltPayload = 3,
        GlobalHist = 4,
        PassHist = 5,
        Index = 6,
    };

    class InitOneSweep : public ComputeKernelBase
    {
    public:
        InitOneSweep(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"InitOneSweep",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& globalHist,
            const D3D12_GPU_VIRTUAL_ADDRESS& passHist,
            const D3D12_GPU_VIRTUAL_ADDRESS& index,
            const uint32_t& threadBlocks)
        {
            std::array<uint32_t, 4> t = { 0, 0, threadBlocks, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, globalHist);
            cmdList->SetComputeRootUnorderedAccessView(2, passHist);
            cmdList->SetComputeRootUnorderedAccessView(3, index);
            cmdList->Dispatch(256, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::PassHist);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::Index);
            return rootParams;
        }
    };

    class GlobalHist : public ComputeKernelBase
    {
    public:
        explicit GlobalHist(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"GlobalHistogram",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& globalHist,
            const uint32_t& numKeys,
            const uint32_t& threadBlocks)
        {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = {
                    numKeys,
                    0,
                    threadBlocks,
                    k_isNotPartialBitFlag };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                numKeys,
                0,
                threadBlocks,
                fullBlocks << 1 | k_isPartialBitFlag };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
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
            const D3D12_GPU_VIRTUAL_ADDRESS& globalHist,
            const D3D12_GPU_VIRTUAL_ADDRESS& passHist,
            const uint32_t& threadBlocks,
            const uint32_t& radixPasses)
        {
            std::array<uint32_t, 4> t = { 0, 0, threadBlocks, 0 };
            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, globalHist);
            cmdList->SetComputeRootUnorderedAccessView(2, passHist);
            cmdList->Dispatch(radixPasses, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::GlobalHist);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::PassHist);
            return rootParams;
        }
    };

    class DigitBinningPass : public ComputeKernelBase
    {
    public:
        explicit DigitBinningPass(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"DigitBinningPass",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& altBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& sortPayloadBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& altPayloadBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& index,
            winrt::com_ptr<ID3D12Resource> passHist,
            const uint32_t& numKeys,
            const uint32_t& threadBlocks,
            const uint32_t& radixShift)
        {
            const uint32_t fullBlocks = threadBlocks / k_maxDim;

            //Setting the partition flag here is unnecessary, because
            //we atomically assign partition tiles
            if (fullBlocks)
            {
                std::array<uint32_t, 4> t = { 
                    numKeys,
                    radixShift,
                    threadBlocks,
                    0 };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, altBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, sortPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(4, altPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(5, passHist->GetGPUVirtualAddress());
                cmdList->SetComputeRootUnorderedAccessView(6, index);
                cmdList->Dispatch(k_maxDim, fullBlocks, 1);

                //To be absolutely safe, add a barrier here on the pass histogram
                //As threadblocks in the second dispatch are dependent on the first dispatch
                UAVBarrierSingle(cmdList, passHist);
            }

            const uint32_t partialBlocks = threadBlocks - fullBlocks * k_maxDim;
            if (partialBlocks)
            {
                std::array<uint32_t, 4> t = {
                    numKeys,
                    radixShift,
                    threadBlocks,
                    0 };

                SetPipelineState(cmdList);
                cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
                cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
                cmdList->SetComputeRootUnorderedAccessView(2, altBuffer);
                cmdList->SetComputeRootUnorderedAccessView(3, sortPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(4, altPayloadBuffer);
                cmdList->SetComputeRootUnorderedAccessView(5, passHist->GetGPUVirtualAddress());
                cmdList->SetComputeRootUnorderedAccessView(6, index);
                cmdList->Dispatch(partialBlocks, 1, 1);
            }
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(7);
            rootParams[0].InitAsConstants(4, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::Alt);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::SortPayload);
            rootParams[4].InitAsUnorderedAccessView((UINT)Reg::AltPayload);
            rootParams[5].InitAsUnorderedAccessView((UINT)Reg::PassHist);
            rootParams[6].InitAsUnorderedAccessView((UINT)Reg::Index);
            return rootParams;
        }
    };
}