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
#include "GPUSorting.h"
#include "GPUSortBase.h"
#include "FFXParallelSortKernels.h"

class FFXParallelSort : public GPUSortBase
{
    uint32_t m_numReduceBlocks;

    uint32_t k_maxThreadGroupsToRun;

	winrt::com_ptr<ID3D12Resource> m_sumTableBuffer;
	winrt::com_ptr<ID3D12Resource> m_reduceTableBuffer;

	FFXParallelSortKernels::FfxPsCount* m_psCount;
	FFXParallelSortKernels::FfxPsCountReduce* m_psCountReduce;
	FFXParallelSortKernels::FfxPsScan* m_psScan;
	FFXParallelSortKernels::FfxPsScanAdd* m_psScanAdd;
	FFXParallelSortKernels::FfxPsScatter* m_psScatter;

public:
    FFXParallelSort(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType);

    FFXParallelSort(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType,
        GPUSorting::PAYLOAD_TYPE payloadType);

    ~FFXParallelSort();

protected:
    void SetCompileArguments() override;

    void InitComputeShaders() override;

    void UpdateSize(uint32_t size) override;

    void DisposeBuffers() override;

    void InitStaticBuffers() override;

    void InitBuffers(
        const uint32_t numKeys,
        const uint32_t threadBlocks) override;

    void PrepareSortCmdList() override;
};