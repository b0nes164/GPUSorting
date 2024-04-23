/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/23/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#include "pch.h"
#include "ForwardSweep.h"

ForwardSweep::ForwardSweep(
	winrt::com_ptr<ID3D12Device> _device,
	GPUSorting::DeviceInfo _deviceInfo,
	GPUSorting::ORDER sortingOrder,
	GPUSorting::KEY_TYPE keyType) :
    SweepBase(
        _device,
        _deviceInfo,
        sortingOrder,
        keyType,
        "ForwardSweep ",
        4,
        256,
        1 << 13)
{
    m_device.copy_from(_device.get());
    SetCompileArguments();
    Initialize();
}

ForwardSweep::ForwardSweep(
    winrt::com_ptr<ID3D12Device> _device,
    GPUSorting::DeviceInfo _deviceInfo,
    GPUSorting::ORDER sortingOrder,
    GPUSorting::KEY_TYPE keyType,
    GPUSorting::PAYLOAD_TYPE payloadType) :
    SweepBase(
        _device,
        _deviceInfo,
        sortingOrder,
        keyType,
        payloadType,
        "ForwardSweep ",
        4,
        256,
        1 << 13)
{
    m_device.copy_from(_device.get());
    SetCompileArguments();
    Initialize();
}

ForwardSweep::~ForwardSweep()
{
}

void ForwardSweep::InitComputeShaders()
{
    const std::filesystem::path path = "Shaders/ForwardSweep.hlsl";
    m_initSweep = new SweepCommonKernels::InitSweep(m_device, m_devInfo, m_compileArguments, path);
    m_globalHist = new SweepCommonKernels::GlobalHist(m_device, m_devInfo, m_compileArguments, path);
    m_scan = new SweepCommonKernels::Scan(m_device, m_devInfo, m_compileArguments, path);
    m_digitPass = new SweepCommonKernels::DigitBinningPass(m_device, m_devInfo, m_compileArguments, path, L"ForwardSweep");
}
