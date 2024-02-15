#pragma once
#include "pch.h"
#include "DeviceRadixSortKernels.h"
#include "Utils.h"

class DeviceRadixSort
{
	const uint32_t radixPasses = 4;
	const uint32_t radix = 256;
	const uint32_t partitionSize = 3840;
	const uint32_t maxErrorReadback = 1024;

	uint32_t numKeys = 0;
	uint32_t partitions = 0;

	winrt::com_ptr<ID3D12Device> m_device;
	DeviceInfo m_devInfo{};

	winrt::com_ptr<ID3D12GraphicsCommandList> m_cmdList;
	winrt::com_ptr<ID3D12CommandQueue> m_cmdQueue;
	winrt::com_ptr<ID3D12CommandAllocator> m_cmdAllocator;

	winrt::com_ptr<ID3D12QueryHeap> m_queryHeap;
	winrt::com_ptr<ID3D12Fence> m_fence;
	wil::unique_event_nothrow m_fenceEvent;
	uint64_t m_nextFenceValue;
	uint64_t m_timestampFrequency;

	winrt::com_ptr<ID3D12Resource> m_sortBuffer;
	winrt::com_ptr<ID3D12Resource> m_sortPayloadBuffer;
	winrt::com_ptr<ID3D12Resource> m_altBuffer;
	winrt::com_ptr<ID3D12Resource> m_altPayloadBuffer;
	winrt::com_ptr<ID3D12Resource> m_passHistBuffer;
	winrt::com_ptr<ID3D12Resource> m_globalHistBuffer;
	winrt::com_ptr<ID3D12Resource> m_errorCountBuffer;
	winrt::com_ptr<ID3D12Resource> m_errorBuffer;
	winrt::com_ptr<ID3D12Resource> m_readBackBuffer;

	InitDeviceRadixSort* m_initDeviceRadix;
	Upsweep* m_upsweep;
	Scan* m_scan;
	Downsweep* m_downsweep;
	InitSortInput* m_initSortInput;
	ClearErrorCount* m_clearErrorCount;
	Validate* m_validate;
	InitScanTestValues* m_initScanTestValues;
	
public:
	DeviceRadixSort(winrt::com_ptr<ID3D12Device> _device, DeviceInfo _deviceInfo);

	void TestSort(uint32_t testSize, uint32_t seed, bool shouldReadBack, bool shouldValidate);

	void TestSortPayload(uint32_t testSize, uint32_t seed, bool shouldReadBack, bool shouldValidate);

	void BatchTiming(uint32_t inputSize, uint32_t batchSize);

	void TestAll();

private:
	bool ValidateScan(uint32_t size);

	bool ValidateOutput(winrt::com_ptr<ID3D12Resource> toValidate, bool shouldPrint, const char* whatValidated);

	bool ValidateSortAndPayload(uint32_t seed);

	double TimeSort(uint32_t seed);

	void UpdateSize(uint32_t size);

	void CreateTestInput(uint32_t seed);

	void PrepareSortCmdList();

	void InitBuffers(const uint32_t& numKeys, const uint32_t& radixPasses, const uint32_t radixDigits, const uint32_t threadBlocks);
};