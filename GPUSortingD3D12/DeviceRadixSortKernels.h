#pragma once
#include "pch.h"
#include "ComputeShader.h"
#include "SharedTypes.h"

enum class Register
{
	Sort = 0,
	SortPayload = 1,
	Alt = 2,
	AltPayload = 3,
	GlobalHist = 4,
	PassHist = 5,
	ErrorCount = 6,
	Error = 7,
};

class InitDeviceRadixSort
{
	ComputeShader* shader;
public:
	InitDeviceRadixSort(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(1);
		rootParameters[0].InitAsUnorderedAccessView((UINT)Register::GlobalHist);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
			L"InitDeviceRadixSort",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS globalHist,
		const uint32_t& threadBlocks)
	{
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRootUnorderedAccessView(0, globalHist);
		cmdList->Dispatch(threadBlocks, 1, 1);
	}
};

class Upsweep
{
	ComputeShader* shader;
public:
	explicit Upsweep(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
		rootParameters[0].InitAsConstants(4, 0);
		rootParameters[1].InitAsUnorderedAccessView((UINT)Register::Sort);
		rootParameters[2].InitAsUnorderedAccessView((UINT)Register::GlobalHist);
		rootParameters[3].InitAsUnorderedAccessView((UINT)Register::PassHist);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
			L"Upsweep",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS globalHist,
		D3D12_GPU_VIRTUAL_ADDRESS passHist,
		const uint32_t& numKeys,
		const uint32_t& threadBlocks,
		const uint32_t& radixShift)
	{

		std::array<uint32_t, 4> t = { numKeys, radixShift, threadBlocks, 0 };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants( 0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
		cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
		cmdList->SetComputeRootUnorderedAccessView(3, passHist);
		cmdList->Dispatch(threadBlocks, 1, 1);
	}
};

class Scan
{
	ComputeShader* shader;
public:
	explicit Scan(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(2);
		rootParameters[0].InitAsConstants(4, 0);
		rootParameters[1].InitAsUnorderedAccessView((UINT)Register::PassHist);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
			L"Scan",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS passHist,
		const uint32_t& partitions)
	{
		std::array<uint32_t, 4> t = { 0, 0, partitions, 0 };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, passHist);
		cmdList->Dispatch(256, 1, 1);
	}
};

class Downsweep
{
	ComputeShader* shader;
public:
	explicit Downsweep(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(7);
		rootParameters[0].InitAsConstants(4, 0);
		rootParameters[1].InitAsUnorderedAccessView((UINT)Register::Sort);
		rootParameters[2].InitAsUnorderedAccessView((UINT)Register::SortPayload);
		rootParameters[3].InitAsUnorderedAccessView((UINT)Register::Alt);
		rootParameters[4].InitAsUnorderedAccessView((UINT)Register::AltPayload);
		rootParameters[5].InitAsUnorderedAccessView((UINT)Register::GlobalHist);
		rootParameters[6].InitAsUnorderedAccessView((UINT)Register::PassHist);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
			L"Downsweep",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS sortPayloadBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS altBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS altPayloadBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS globalHist,
		D3D12_GPU_VIRTUAL_ADDRESS passHist,
		const uint32_t& numKeys,
		const uint32_t& threadBlocks,
		const uint32_t& radixShift)
	{
		std::array<uint32_t, 4> t = { numKeys, radixShift, threadBlocks, 0 };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
		cmdList->SetComputeRootUnorderedAccessView(2, sortPayloadBuffer);
		cmdList->SetComputeRootUnorderedAccessView(3, altBuffer);
		cmdList->SetComputeRootUnorderedAccessView(4, altPayloadBuffer);
		cmdList->SetComputeRootUnorderedAccessView(5, globalHist);
		cmdList->SetComputeRootUnorderedAccessView(6, passHist);
		cmdList->Dispatch(threadBlocks, 1, 1);
	}
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
		rootParameters[1].InitAsUnorderedAccessView((UINT)Register::Sort);
		rootParameters[2].InitAsUnorderedAccessView((UINT)Register::SortPayload);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
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
		std::array<uint32_t, 4> t = { numKeys, 0, 0, seed };
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
		rootParameters[0].InitAsUnorderedAccessView((UINT)Register::ErrorCount);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
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
		rootParameters[1].InitAsUnorderedAccessView((UINT)Register::Sort);
		rootParameters[2].InitAsUnorderedAccessView((UINT)Register::ErrorCount);
		rootParameters[3].InitAsUnorderedAccessView((UINT)Register::Error);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
			L"Validate",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS errorCount,
		D3D12_GPU_VIRTUAL_ADDRESS error,
		const uint32_t& numKeys,
		const uint32_t& maxErrors)
	{
		uint32_t valThreadBlocks = (numKeys + valPartSize - 1) / valPartSize;
		std::array<uint32_t, 4> t = { numKeys, 0, valThreadBlocks, maxErrors };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
		cmdList->SetComputeRootUnorderedAccessView(2, errorCount);
		cmdList->SetComputeRootUnorderedAccessView(3, error);
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
		rootParameters[1].InitAsUnorderedAccessView((UINT)Register::PassHist);

		shader = new ComputeShader(
			device,
			info,
			"DeviceRadixSort.hlsl",
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