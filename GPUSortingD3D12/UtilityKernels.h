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
#include "GPUSorting.h"

enum class UtilReg
{
    Sort = 0,
    SortPayload = 1,
    PassHist = 2,
    ErrorCount = 3,
};

class InitSortInput
{
    ComputeShader* shader;
public:

    explicit InitSortInput(
        winrt::com_ptr<ID3D12Device> device,
        DeviceInfo const& info,
        std::vector<std::wstring> compileArguments)
    {
        auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
        rootParameters[0].InitAsConstants(4, 0);
        rootParameters[1].InitAsUnorderedAccessView((UINT)UtilReg::Sort);
        rootParameters[2].InitAsUnorderedAccessView((UINT)UtilReg::SortPayload);

        shader = new ComputeShader(
            device,
            info,
            "Shaders/Utility.hlsl",
            L"InitSortInput",
            compileArguments,
            rootParameters);
    }

    void Dispatch(
        winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
        D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS sortPayloadBuffer,
        const uint32_t& numKeys,
        uint32_t seed)
    {
        std::array<uint32_t, 4> t = { numKeys, 0, seed, 0 };
        shader->SetPipelineState(cmdList);
        cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
        cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
        cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
        cmdList->Dispatch(256, 1, 1);
    }
};

class ClearErrorCount
{
    ComputeShader* shader;
public:
    explicit ClearErrorCount(
        winrt::com_ptr<ID3D12Device> device,
        DeviceInfo const& info,
        std::vector<std::wstring> compileArguments)
    {
        auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
        rootParameters[0].InitAsUnorderedAccessView((UINT)UtilReg::ErrorCount);

        shader = new ComputeShader(
            device,
            info,
            "Shaders/Utility.hlsl",
            L"ClearErrorCount",
            compileArguments,
            rootParameters);
    }

    void Dispatch(
        winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
        D3D12_GPU_VIRTUAL_ADDRESS errorCount)
    {
        shader->SetPipelineState(cmdList);
        cmdList->SetComputeRootUnorderedAccessView(0, errorCount);
        cmdList->Dispatch(1, 1, 1);
    }
};

class Validate
{
    const uint32_t valPartSize = 2048;
    ComputeShader* shader;
public:
    explicit Validate(
        winrt::com_ptr<ID3D12Device> device,
        DeviceInfo const& info,
        std::vector<std::wstring> compileArguments)
    {
        auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
        rootParameters[0].InitAsConstants(4, 0);
        rootParameters[1].InitAsUnorderedAccessView((UINT)UtilReg::Sort);
        rootParameters[2].InitAsUnorderedAccessView((UINT)UtilReg::SortPayload);
        rootParameters[3].InitAsUnorderedAccessView((UINT)UtilReg::ErrorCount);

        shader = new ComputeShader(
            device,
            info,
            "Shaders/Utility.hlsl",
            L"Validate",
            compileArguments,
            rootParameters);
    }

    void Dispatch(
        winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
        D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS sortPayloadBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS errorCount,
        const uint32_t& numKeys)
    {
        uint32_t valThreadBlocks = (numKeys + valPartSize - 1) / valPartSize;
        std::array<uint32_t, 4> t = { numKeys, valThreadBlocks, 0, 0 };
        shader->SetPipelineState(cmdList);
        cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
        cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
        cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
        cmdList->SetComputeRootUnorderedAccessView(3, errorCount);
        cmdList->Dispatch(valThreadBlocks, 1, 1);
    }
};

class InitScanTestValues
{
    ComputeShader* shader;
public:
    explicit InitScanTestValues(
        winrt::com_ptr<ID3D12Device> device,
        DeviceInfo const& info,
        std::vector<std::wstring> compileArguments)
    {
        auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
        rootParameters[0].InitAsConstants(4, 0);
        rootParameters[1].InitAsUnorderedAccessView((UINT)UtilReg::PassHist);

        shader = new ComputeShader(
            device,
            info,
            "Shaders/Utility.hlsl",
            L"InitScanTestValues",
            compileArguments,
            rootParameters);
    }

    void Dispatch(
        winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
        D3D12_GPU_VIRTUAL_ADDRESS passHist,
        const uint32_t& numKeys)
    {
        std::array<uint32_t, 4> t = { numKeys, 0, 0, 0 };
        shader->SetPipelineState(cmdList);
        cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
        cmdList->SetComputeRootUnorderedAccessView(1, passHist);
        cmdList->Dispatch(1, 1, 1);
    }
};