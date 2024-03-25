/******************************************************************************
 * FFXParllelSort
 * This algorithm is part of the FidelityFX SDK.
 * https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "ComputeKernelBase.h"

namespace FFXParallelSortKernels
{
    enum class Reg
    {
        SrcBuffer = 0,
        SrcPayload = 1,
        SumTable = 2,
        ReduceTable = 3,
        DstBuffer = 4,
        DstPayload = 5,
        ScanSrc = 6,
        ScanDst = 7,
        ScanScratch = 8,
    };

    class FfxPsCount : public ComputeKernelBase
    {
    public:
        FfxPsCount(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"FPS_Count",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            const D3D12_GPU_VIRTUAL_ADDRESS& srcBuffer,
            const D3D12_GPU_VIRTUAL_ADDRESS& sumTable,
            const uint32_t& numKeys,
            const uint32_t& numThreadGroups,
            const uint32_t& numBlocksPerThreadGroup,
            const uint32_t& numThreadGroupsWithAdditionalBlocks,
            const uint32_t& radixShift)
        {
            std::array<uint32_t, 8> t = {
                numKeys,
                numThreadGroups,
                numBlocksPerThreadGroup,
                numThreadGroupsWithAdditionalBlocks,
                0,
                0,
                radixShift,
                0 };

            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, srcBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, sumTable);
            cmdList->Dispatch(numThreadGroups, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParams[0].InitAsConstants(8, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::SrcBuffer);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::SumTable);
            return rootParams;
        }
    };

    class FfxPsCountReduce : public ComputeKernelBase
    {
    public:
        FfxPsCountReduce(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"FPS_CountReduce",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS sumTable,
            D3D12_GPU_VIRTUAL_ADDRESS reduceTable,
            const uint32_t& numThreadGroups,
            const uint32_t& numScanValues,
            const uint32_t& numReduceThreadGroupPerBin)
        {
            std::array<uint32_t, 8> t = {
                0,
                numThreadGroups,
                0,
                0,
                numReduceThreadGroupPerBin,
                0,
                0,
                0 };

            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, sumTable);
            cmdList->SetComputeRootUnorderedAccessView(2, reduceTable);
            cmdList->Dispatch(numScanValues, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
            rootParams[0].InitAsConstants(8, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::SumTable);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::ReduceTable);
            return rootParams;
        }
    };

    class FfxPsScan : public ComputeKernelBase
    {
    public:
        FfxPsScan(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"FPS_Scan",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanSrc,
            D3D12_GPU_VIRTUAL_ADDRESS scanDst,
            D3D12_GPU_VIRTUAL_ADDRESS scanScratch,
            const uint32_t& numThreadGroups,
            const uint32_t& numScanValues)
        {
            std::array<uint32_t, 8> t = {
                0,
                0,
                0,
                0,
                0,
                numScanValues,
                0,
                0 };

            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanSrc);
            cmdList->SetComputeRootUnorderedAccessView(2, scanDst);
            cmdList->SetComputeRootUnorderedAccessView(3, scanScratch);
            cmdList->Dispatch(numThreadGroups, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParams[0].InitAsConstants(8, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::ScanSrc);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::ScanDst);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::ScanScratch);
            return rootParams;
        }
    };

    class FfxPsScanAdd : public ComputeKernelBase
    {
    public:
        FfxPsScanAdd(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"FPS_ScanAdd",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS scanSrc,
            D3D12_GPU_VIRTUAL_ADDRESS scanDst,
            D3D12_GPU_VIRTUAL_ADDRESS scanScratch,
            const uint32_t& numThreadGroups,
            const uint32_t& numScanValues,
            const uint32_t& numReduceThreadGroupPerBin)
        {
            std::array<uint32_t, 8> t = {
                0,
                numThreadGroups,
                0,
                0,
                numReduceThreadGroupPerBin,
                0,
                0,
                0 };

            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, scanSrc);
            cmdList->SetComputeRootUnorderedAccessView(2, scanDst);
            cmdList->SetComputeRootUnorderedAccessView(3, scanScratch);
            cmdList->Dispatch(numScanValues, 1, 1);
        }

    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
            rootParams[0].InitAsConstants(8, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::ScanSrc);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::ScanDst);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::ScanScratch);
            return rootParams;
        }
    };

    class FfxPsScatter : public ComputeKernelBase
    {
    public:
        FfxPsScatter(
            winrt::com_ptr<ID3D12Device> device,
            const GPUSorting::DeviceInfo& info,
            const std::vector<std::wstring>& compileArguments,
            const std::filesystem::path& shaderPath) :
            ComputeKernelBase(
                device,
                info,
                shaderPath,
                L"FPS_Scatter",
                compileArguments,
                CreateRootParameters())
        {
        }

        void Dispatch(
            winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
            D3D12_GPU_VIRTUAL_ADDRESS srcBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS srcPayload,
            D3D12_GPU_VIRTUAL_ADDRESS dstBuffer,
            D3D12_GPU_VIRTUAL_ADDRESS dstPayload,
            D3D12_GPU_VIRTUAL_ADDRESS sumTable,
            const uint32_t& numKeys,
            const uint32_t& numThreadGroups,
            const uint32_t& numBlocksPerThreadGroup,
            const uint32_t& numThreadGroupsWithAdditionalBlocks,
            const uint32_t& radixShift)
        {
            std::array<uint32_t, 8> t = {
                numKeys,
                numThreadGroups,
                numBlocksPerThreadGroup,
                numThreadGroupsWithAdditionalBlocks,
                0,
                0,
                radixShift,
                0 };

            SetPipelineState(cmdList);
            cmdList->SetComputeRoot32BitConstants(0, (uint32_t)t.size(), t.data(), 0);
            cmdList->SetComputeRootUnorderedAccessView(1, srcBuffer);
            cmdList->SetComputeRootUnorderedAccessView(2, srcPayload);
            cmdList->SetComputeRootUnorderedAccessView(3, dstBuffer);
            cmdList->SetComputeRootUnorderedAccessView(4, dstPayload);
            cmdList->SetComputeRootUnorderedAccessView(5, sumTable);
            cmdList->Dispatch(numThreadGroups, 1, 1);
        }
    protected:
        const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() override
        {
            auto rootParams = std::vector<CD3DX12_ROOT_PARAMETER1>(6);
            rootParams[0].InitAsConstants(8, 0);
            rootParams[1].InitAsUnorderedAccessView((UINT)Reg::SrcBuffer);
            rootParams[2].InitAsUnorderedAccessView((UINT)Reg::SrcPayload);
            rootParams[3].InitAsUnorderedAccessView((UINT)Reg::DstBuffer);
            rootParams[4].InitAsUnorderedAccessView((UINT)Reg::DstPayload);
            rootParams[5].InitAsUnorderedAccessView((UINT)Reg::SumTable);
            return rootParams;
        }
    };
}