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
#include "GPUSorting.h"
#include "GPUSorter.h"
#include "OneSweepKernels.h"

class OneSweep : public GPUSorter
{
	winrt::com_ptr<ID3D12Resource> m_indexBuffer;
	winrt::com_ptr<ID3D12Resource> m_passHistBuffer;
	winrt::com_ptr<ID3D12Resource> m_globalHistBuffer;

	InitOneSweep* m_initOneSweep;
	GlobalHist* m_globalHist;
	DigitBinningPass* m_digitBinningPass;

public:
	OneSweep(
		winrt::com_ptr<ID3D12Device> _device,
		DeviceInfo _deviceInfo,
		GPU_SORTING_ORDER sortingOrder,
		GPU_SORTING_KEY_TYPE keyType);

	OneSweep(
		winrt::com_ptr<ID3D12Device> _device,
		DeviceInfo _deviceInfo,
		GPU_SORTING_ORDER sortingOrder,
		GPU_SORTING_KEY_TYPE keyType,
		GPU_SORTING_PAYLOAD_TYPE payloadType);

	void TestAll() override;

protected:
	void InitComputeShaders() override;

	void DisposeBuffers() override;

	void InitStaticBuffers() override;

	void InitBuffers(
		const uint32_t numKeys,
		const uint32_t threadBlocks) override;

	void PrepareSortCmdList() override;
};