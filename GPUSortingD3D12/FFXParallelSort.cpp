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
#include "pch.h"
#include "FFXParallelSort.h"

FFXParallelSort::FFXParallelSort(
    winrt::com_ptr<ID3D12Device> _device,
    GPUSorting::DeviceInfo _deviceInfo,
    GPUSorting::ORDER sortingOrder,
    GPUSorting::KEY_TYPE keyType) :
    GPUSortBase(
        _device,
        _deviceInfo,
        sortingOrder,
        GPUSorting::KEY_UINT32,
        "FFXParallelSort ",
        8,
        16,
        1 << 13,
        GPUSorting::TuningParameters{ false, 4, 256, 1024, 0 }),  //Pass in a set of parameters to match FFX
    k_maxThreadGroupsToRun(1024)
{
    m_device.copy_from(_device.get());

    if (keyType != GPUSorting::KEY_UINT32)
    {
        printf("\nWarning, FFXParallelSort implementation only supports uint32_t for keys.\n");
        printf("Sort object initialized as uin32_t.\n");
    }

    if (sortingOrder != GPUSorting::ORDER_ASCENDING)
    {
        printf("\nWarning, FFXParallelSort implementation only supports ascending order sorting.\n");
        printf("Sort object initialized as ascending.\n");
    }

    SetCompileArguments();
    Initialize();
}

FFXParallelSort::FFXParallelSort(
    winrt::com_ptr<ID3D12Device> _device,
    GPUSorting::DeviceInfo _deviceInfo,
    GPUSorting::ORDER sortingOrder,
    GPUSorting::KEY_TYPE keyType,
    GPUSorting::PAYLOAD_TYPE payloadType) :
    GPUSortBase(
        _device,
        _deviceInfo,
        sortingOrder,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32,
        "FFXParallelSort ",
        8,
        16,
        1 << 13,
        GPUSorting::TuningParameters{ false, 4, 256, 1024, 0 }),  //Pass in a set of parameters to match FFX
    k_maxThreadGroupsToRun(1024)
{
    m_device.copy_from(_device.get());

    if (keyType != GPUSorting::KEY_UINT32)
    {
        printf("\nWarning, FFXParallelSort implementation only supports uint32_t for keys.\n");
        printf("Sort object initialized as uin32_t.\n");
    }

    if (payloadType != GPUSorting::PAYLOAD_UINT32)
    {
        printf("\nWarning, FFXParallelSort implementation only supports uint32_t for payloads.\n");
        printf("Sort object initialized as uin32_t.\n");
    }

    if (sortingOrder != GPUSorting::ORDER_ASCENDING)
    {
        printf("\nWarning, FFXParallelSort implementation only supports ascending order sorting.\n");
        printf("Sort object initialized as ascending.\n");
    }
    SetCompileArguments();
    Initialize();
}

FFXParallelSort::~FFXParallelSort()
{
}

//FFX does not include the large size partition test
//as it uses a different method to handle large partitions
bool FFXParallelSort::TestAll()
{
    printf("Beginning ");
    printf(k_sortName);
    PrintSortingConfig(k_sortingConfig);
    printf("test all. \n");

    uint32_t sortPayloadTestsPassed = 0;
    const uint32_t testEnd = k_tuningParameters.partitionSize * 2 + 1;
    for (uint32_t i = k_tuningParameters.partitionSize; i < testEnd; ++i)
    {
        sortPayloadTestsPassed += ValidateSort(i, i);

        if (!(i & 127))
            printf(".");
    }

    printf("\n");
    printf("%u / %u passed. \n", sortPayloadTestsPassed, k_tuningParameters.partitionSize + 1);

    printf("Beginning large size tests\n");
    sortPayloadTestsPassed += ValidateSort(1 << 21, 5);
    sortPayloadTestsPassed += ValidateSort(1 << 22, 7);
    sortPayloadTestsPassed += ValidateSort(1 << 23, 11);

    uint32_t totalTests = k_tuningParameters.partitionSize + 1 + 6;
    if (sortPayloadTestsPassed == totalTests)
    {
        printf("%u / %u  All tests passed. \n\n", totalTests, totalTests);
        return true;
    }
    else
    {
        printf("%u / %u  Test failed. \n\n", sortPayloadTestsPassed, totalTests);
        return false;
    }

    return sortPayloadTestsPassed == totalTests;
}

void FFXParallelSort::SetCompileArguments()
{
    m_compileArguments.push_back(L"-DSHOULD_ASCEND");
    m_compileArguments.push_back(L"-DKEY_UINT");

    if (k_sortingConfig.sortingMode == GPUSorting::MODE_PAIRS)
    {
        m_compileArguments.push_back(L"-DFFX_PARALLELSORT_COPY_VALUE");
        m_compileArguments.push_back(L"-DSORT_PAIRS");
        m_compileArguments.push_back(L"-DPAYLOAD_UINT");
    }

    m_compileArguments.push_back(L"-O3");
#ifdef _DEBUG
    m_compileArguments.push_back(L"-Zi");
#endif
}

void FFXParallelSort::InitComputeShaders()
{
    const std::filesystem::path path = "Shaders/FFXParallelSort.hlsl";
    m_psCount = new FFXParallelSortKernels::FfxPsCount(m_device, m_devInfo, m_compileArguments, path);
    m_psCountReduce = new FFXParallelSortKernels::FfxPsCountReduce(m_device, m_devInfo, m_compileArguments, path);
    m_psScan = new FFXParallelSortKernels::FfxPsScan(m_device, m_devInfo, m_compileArguments, path);
    m_psScanAdd = new FFXParallelSortKernels::FfxPsScanAdd(m_device, m_devInfo, m_compileArguments, path);
    m_psScatter = new FFXParallelSortKernels::FfxPsScatter(m_device, m_devInfo, m_compileArguments, path);
}

void FFXParallelSort::UpdateSize(uint32_t size)
{
    if (m_numKeys != size)
    {
        m_numKeys = size;
        m_partitions = divRoundUp(m_numKeys, k_tuningParameters.partitionSize);
        m_numReduceBlocks = divRoundUp(m_partitions, k_tuningParameters.partitionSize);

        DisposeBuffers();
        InitBuffers(m_numKeys, m_partitions);
    }
}

void FFXParallelSort::DisposeBuffers()
{
    m_sortBuffer = nullptr;
    m_sortPayloadBuffer = nullptr;
    m_altBuffer = nullptr;
    m_altPayloadBuffer = nullptr;
    m_sumTableBuffer = nullptr;
    m_reduceTableBuffer = nullptr;
}

void FFXParallelSort::InitStaticBuffers()
{
    m_errorCountBuffer = CreateBuffer(
        m_device,
        1 * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_readBackBuffer = CreateBuffer(
        m_device,
        k_maxReadBack * sizeof(uint32_t),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE);
}

//TO DO: This could be better. Too bad!
void FFXParallelSort::InitBuffers(const uint32_t numKeys, const uint32_t threadBlocks)
{
    m_sortBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_altBuffer = CreateBuffer(
        m_device,
        numKeys * sizeof(uint32_t),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_sumTableBuffer = CreateBuffer(
        m_device,
        threadBlocks * k_radix,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    m_reduceTableBuffer = CreateBuffer(
        m_device,
        m_numReduceBlocks * k_radix,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    if (k_sortingConfig.sortingMode == GPUSorting::MODE_PAIRS)
    {
        m_sortPayloadBuffer = CreateBuffer(
            m_device,
            numKeys * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        m_altPayloadBuffer = CreateBuffer(
            m_device,
            numKeys * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    else
    {
        m_sortPayloadBuffer = CreateBuffer(
            m_device,
            1 * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        m_altPayloadBuffer = CreateBuffer(
            m_device,
            1 * sizeof(uint32_t),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
}

void FFXParallelSort::PrepareSortCmdList()
{
    uint32_t numThreadGroupsToRun;
    uint32_t numBlocksPerThreadGroup;
    uint32_t numThreadGroupsWithAdditionalBlocks;
    uint32_t numScanValues;
    uint32_t numReduceThreadGroupPerBin;

    if (m_partitions < k_maxThreadGroupsToRun)
    {
        numThreadGroupsToRun = m_partitions;
        numBlocksPerThreadGroup = 1;
        numThreadGroupsWithAdditionalBlocks = 0;
    }
    else
    {
        numThreadGroupsToRun = k_maxThreadGroupsToRun;
        numBlocksPerThreadGroup = m_partitions / k_maxThreadGroupsToRun;
        numThreadGroupsWithAdditionalBlocks = m_partitions % k_maxThreadGroupsToRun;
    }

    numScanValues = k_radix * ((k_tuningParameters.partitionSize > numThreadGroupsToRun) ?
        1 : divRoundUp(numThreadGroupsToRun, k_tuningParameters.partitionSize));
    numReduceThreadGroupPerBin = numScanValues / k_radix;

    for (uint32_t radixShift = 0; radixShift < 32; radixShift += 4)
    {
        m_psCount->Dispatch(
            m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_sumTableBuffer->GetGPUVirtualAddress(),
            m_numKeys,
            numThreadGroupsToRun,
            numBlocksPerThreadGroup,
            numThreadGroupsWithAdditionalBlocks,
            radixShift);
        UAVBarrierSingle(m_cmdList, m_sumTableBuffer);

        m_psCountReduce->Dispatch(
            m_cmdList,
            m_sumTableBuffer->GetGPUVirtualAddress(),
            m_reduceTableBuffer->GetGPUVirtualAddress(),
            numThreadGroupsToRun,
            numScanValues,
            numReduceThreadGroupPerBin);
        UAVBarrierSingle(m_cmdList, m_reduceTableBuffer);

        m_psScan->Dispatch(
            m_cmdList,
            m_reduceTableBuffer->GetGPUVirtualAddress(),
            m_reduceTableBuffer->GetGPUVirtualAddress(),
            m_sumTableBuffer->GetGPUVirtualAddress(),
            1,
            numScanValues);
        UAVBarrierSingle(m_cmdList, m_reduceTableBuffer);

        m_psScanAdd->Dispatch(
            m_cmdList,
            m_sumTableBuffer->GetGPUVirtualAddress(),
            m_sumTableBuffer->GetGPUVirtualAddress(),
            m_reduceTableBuffer->GetGPUVirtualAddress(),
            numThreadGroupsToRun,
            numScanValues,
            numReduceThreadGroupPerBin);
        UAVBarrierSingle(m_cmdList, m_reduceTableBuffer);
        UAVBarrierSingle(m_cmdList, m_sumTableBuffer);

        m_psScatter->Dispatch(
            m_cmdList,
            m_sortBuffer->GetGPUVirtualAddress(),
            m_sortPayloadBuffer->GetGPUVirtualAddress(),
            m_altBuffer->GetGPUVirtualAddress(),
            m_altPayloadBuffer->GetGPUVirtualAddress(),
            m_sumTableBuffer->GetGPUVirtualAddress(),
            m_numKeys,
            numThreadGroupsToRun,
            numBlocksPerThreadGroup,
            numThreadGroupsWithAdditionalBlocks,
            radixShift);
        UAVBarrierSingle(m_cmdList, m_sortBuffer);
        UAVBarrierSingle(m_cmdList, m_sortPayloadBuffer);
        UAVBarrierSingle(m_cmdList, m_altBuffer);
        UAVBarrierSingle(m_cmdList, m_altPayloadBuffer);

        swap(m_sortBuffer, m_altBuffer);
        swap(m_sortPayloadBuffer, m_altPayloadBuffer);
    }
}
