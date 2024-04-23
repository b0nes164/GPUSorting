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
}