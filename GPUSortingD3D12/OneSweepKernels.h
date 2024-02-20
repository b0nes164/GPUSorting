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
#include "ComputeShader.h"

enum class OneSweepReg
{
	Sort = 0,
	Alt = 1,
	GlobalHist = 2,
	PassHist = 3,
	Index =  4,
};

class InitOneSweep
{
	ComputeShader* shader;
public:
	InitOneSweep(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(4);
		rootParameters[0].InitAsConstants(4, 0);
		rootParameters[1].InitAsUnorderedAccessView((UINT)OneSweepReg::GlobalHist);
		rootParameters[2].InitAsUnorderedAccessView((UINT)OneSweepReg::PassHist);
		rootParameters[3].InitAsUnorderedAccessView((UINT)OneSweepReg::Index);

		shader = new ComputeShader(
			device,
			info,
			"Shaders/OneSweep.hlsl",
			L"InitOneSweep",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS globalHist,
		D3D12_GPU_VIRTUAL_ADDRESS passHist,
		D3D12_GPU_VIRTUAL_ADDRESS index,
		const uint32_t& threadBlocks)
	{
		std::array<uint32_t, 4> t = { 0, 0, threadBlocks, 0 };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, globalHist);
		cmdList->SetComputeRootUnorderedAccessView(2, passHist);
		cmdList->SetComputeRootUnorderedAccessView(3, index);
		cmdList->Dispatch(256, 1, 1);
	}
};

class GlobalHist
{
	ComputeShader* shader;
public:
	explicit GlobalHist(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(3);
		rootParameters[0].InitAsConstants(4, 0);
		rootParameters[1].InitAsUnorderedAccessView((UINT)OneSweepReg::Sort);
		rootParameters[2].InitAsUnorderedAccessView((UINT)OneSweepReg::GlobalHist);

		shader = new ComputeShader(
			device,
			info,
			"Shaders/OneSweep.hlsl",
			L"GlobalHistogram",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS globalHist,
		const uint32_t& numKeys,
		const uint32_t& threadBlocks)
	{

		std::array<uint32_t, 4> t = { numKeys, 0, threadBlocks, 0 };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
		cmdList->SetComputeRootUnorderedAccessView(2, globalHist);
		cmdList->Dispatch(threadBlocks, 1, 1);
	}
};

class DigitBinningPass
{
	ComputeShader* shader;
public:
	explicit DigitBinningPass(
		winrt::com_ptr<ID3D12Device> device,
		DeviceInfo const& info,
		std::vector<std::wstring> compileArguments)
	{
		auto rootParameters = std::vector<CD3DX12_ROOT_PARAMETER1>(6);
		rootParameters[0].InitAsConstants(4, 0);
		rootParameters[1].InitAsUnorderedAccessView((UINT)OneSweepReg::Sort);
		rootParameters[2].InitAsUnorderedAccessView((UINT)OneSweepReg::Alt);
		rootParameters[3].InitAsUnorderedAccessView((UINT)OneSweepReg::GlobalHist);
		rootParameters[4].InitAsUnorderedAccessView((UINT)OneSweepReg::PassHist);
		rootParameters[5].InitAsUnorderedAccessView((UINT)OneSweepReg::Index);

		shader = new ComputeShader(
			device,
			info,
			"Shaders/OneSweep.hlsl",
			L"DigitBinningPass",
			compileArguments,
			rootParameters);
	}

	void Dispatch(
		winrt::com_ptr<ID3D12GraphicsCommandList> cmdList,
		D3D12_GPU_VIRTUAL_ADDRESS sortBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS altBuffer,
		D3D12_GPU_VIRTUAL_ADDRESS globalHist,
		D3D12_GPU_VIRTUAL_ADDRESS passHist,
		D3D12_GPU_VIRTUAL_ADDRESS index,
		const uint32_t& numKeys,
		const uint32_t& threadBlocks,
		const uint32_t& radixShift)
	{
		std::array<uint32_t, 4> t = { numKeys, radixShift, threadBlocks, 0 };
		shader->SetPipelineState(cmdList);
		cmdList->SetComputeRoot32BitConstants(0, 4, t.data(), 0);
		cmdList->SetComputeRootUnorderedAccessView(1, sortBuffer);
		cmdList->SetComputeRootUnorderedAccessView(2, altBuffer);
		cmdList->SetComputeRootUnorderedAccessView(3, globalHist);
		cmdList->SetComputeRootUnorderedAccessView(4, passHist);
		cmdList->SetComputeRootUnorderedAccessView(5, index);
		cmdList->Dispatch(threadBlocks, 1, 1);
	}
};