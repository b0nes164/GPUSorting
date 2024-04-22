/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/22/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "SweepCommonKernels.h"

namespace EmulatedDeadlockingKernels
{
    class ClearIndex : public ComputeKernelBase
    {
    public:
        ClearIndex(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"ClearIndex",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& index)
        {
            SetPipelineState(cmdList);
            cmdList->SetComputeRootUnorderedAccessView(0, index);
            cmdList->Dispatch(1, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
            rootParams[0].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Index);
            return rootParams;
        }
    };

    class EmulatedDeadlockingPassOne : public ComputeKernelBase
    {
    public:
        EmulatedDeadlockingPassOne(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"EmulatedDeadlockingPassOne",
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
            rootParams[1].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Alt);
            rootParams[3].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::SortPayload);
            rootParams[4].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::AltPayload);
            rootParams[5].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::PassHist);
            rootParams[6].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Index);
            return rootParams;
        }
    };

    class EmulatedDeadlockingPassTwo : public ComputeKernelBase
    {
    public:
        EmulatedDeadlockingPassTwo(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"EmulatedDeadlockingPassTwo",
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
            rootParams[1].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Sort);
            rootParams[2].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Alt);
            rootParams[3].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::SortPayload);
            rootParams[4].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::AltPayload);
            rootParams[5].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::PassHist);
            rootParams[6].InitAsUnorderedAccessView((UINT)SweepCommonKernels::Reg::Index);
            return rootParams;
        }
    };
}