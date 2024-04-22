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
#include "SweepBase.h"
#include "EmulatedDeadlockingKernels.h"

class EmulatedDeadlocking : public SweepBase
{
    EmulatedDeadlockingKernels::ClearIndex* m_clearIndex;
    EmulatedDeadlockingKernels::EmulatedDeadlockingPassOne* m_passOne;
    EmulatedDeadlockingKernels::EmulatedDeadlockingPassTwo* m_passTwo;

public:
    EmulatedDeadlocking(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType);

    EmulatedDeadlocking(
        winrt::com_ptr<ID3D12Device> _device,
        GPUSorting::DeviceInfo _deviceInfo,
        GPUSorting::ORDER sortingOrder,
        GPUSorting::KEY_TYPE keyType,
        GPUSorting::PAYLOAD_TYPE payloadType);

    ~EmulatedDeadlocking();

protected:
    void InitComputeShaders() override;

    void PrepareSortCmdList() override;
};