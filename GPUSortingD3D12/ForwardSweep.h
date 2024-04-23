/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/23/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "SweepBase.h"

class ForwardSweep : public SweepBase
{

public:
    ForwardSweep(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType);

    ForwardSweep(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType,
        GPUSorting::PAYLOAD_TYPE payloadType);

    ~ForwardSweep();

protected:
    void InitComputeShaders() override;
};