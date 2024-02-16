#pragma once
#include "pch.h"
#include "GPUSorting.h"
#include "DeviceRadixSortKernels.h"
#include "UtilityKernels.h"
#include "Utils.h"

class DeviceRadixSort
{
	const uint32_t radixPasses = 4;
	const uint32_t radix = 256;
	const uint32_t partitionSize = 3840;
	const uint32_t maxReadback = 1 << 20;

	uint32_t numKeys = 0;
	uint32_t partitions = 0;

	winrt::com_ptr<ID3D12Device> m_device;
	DeviceInfo m_devInfo{};
	std::vector<std::wstring> m_compileArguments;
	GPU_SORTING_MODE m_sortingMode;

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
	DeviceRadixSort(
		winrt::com_ptr<ID3D12Device> _device, 
		DeviceInfo _deviceInfo,
		GPU_SORTING_ORDER sortingOrder,
		GPU_SORTING_KEY_TYPE keyType);

	DeviceRadixSort(
		winrt::com_ptr<ID3D12Device> _device, 
		DeviceInfo _deviceInfo,
		GPU_SORTING_ORDER sortingOrder,
		GPU_SORTING_KEY_TYPE keyType,
		GPU_SORTING_PAYLOAD_TYPE payloadType);

	void TestSort(
		uint32_t testSize, 
		uint32_t seed, 
		bool shouldReadBack, 
		bool shouldValidate);

	void BatchTiming(uint32_t inputSize, uint32_t batchSize);

	void TestAll();

private:
	void Initialize();

	void UpdateSize(uint32_t size);

	void DisposeBuffers();

	void InitBuffers(const uint32_t numKeys, const uint32_t threadBlocks);

	void InitStaticBuffers();

	void CreateTestInput(uint32_t seed);

	void PrepareSortCmdList();

	void ExecuteCommandList();

	bool ValidateOutput(bool shouldPrint);

	bool ValidateSort(uint32_t size, uint32_t seed);

	bool ValidateScan(uint32_t size);

	double TimeSort(uint32_t seed);
};