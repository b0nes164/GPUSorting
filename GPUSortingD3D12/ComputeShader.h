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
#include "GPUsorting.h"

class ComputeShader
{
public:
    ComputeShader(
        winrt::com_ptr<ID3D12Device> device, 
        DeviceInfo const& info,
        std::filesystem::path const& shaderPath,
        const wchar_t* entryPoint,
        std::vector<std::wstring>& compileArguments,
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters)
    {
        auto byteCode = CompileShader(shaderPath, info, entryPoint,
            compileArguments);
        m_rootSignature = CreateRootSignature(device, rootParameters);

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
        pipelineDesc.pRootSignature = m_rootSignature.get();
        pipelineDesc.CS.pShaderBytecode = byteCode.data();
        pipelineDesc.CS.BytecodeLength = byteCode.size();
        winrt::check_hresult(device->CreateComputePipelineState(
            &pipelineDesc, IID_PPV_ARGS(m_computePipelineStateDesc.put())));
    }

    void SetPipelineState(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList)
    {
        cmdList->SetPipelineState(m_computePipelineStateDesc.get());
        cmdList->SetComputeRootSignature(m_rootSignature.get());
    }

private:
    winrt::com_ptr<ID3D12RootSignature> m_rootSignature;
    winrt::com_ptr<ID3D12PipelineState> m_computePipelineStateDesc;

    std::vector<uint8_t> CompileShader(
        std::filesystem::path const& shaderPath, 
        DeviceInfo const& info, 
        const wchar_t* entryPoint,
        std::vector<std::wstring>& arguments)
    {
        winrt::com_ptr<IDxcUtils> utils;
        winrt::check_hresult(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(utils.put())));

        uint32_t codePage = DXC_CP_UTF8;
        winrt::com_ptr<IDxcBlobEncoding> sourceBlob;
        winrt::check_hresult(utils->LoadFile(
            shaderPath.wstring().c_str(), &codePage, sourceBlob.put()));

        std::vector<wchar_t const*> pargs;
        std::transform(arguments.begin(), arguments.end(), 
            back_inserter(pargs), [](auto const& a) { return a.c_str(); });

        winrt::com_ptr<IDxcCompilerArgs> compilerArgs;
        winrt::check_hresult(utils->BuildArguments(
            shaderPath.wstring().c_str(),
            entryPoint,
            info.SupportedShaderModel.c_str(),
            pargs.data(),
            static_cast<uint32_t>(pargs.size()),
            nullptr,
            0,
            compilerArgs.put()));

        DxcBuffer dxcBuffer = {
            sourceBlob->GetBufferPointer(),
            sourceBlob->GetBufferSize(),
            codePage };

        winrt::com_ptr<IDxcIncludeHandler> includeHandler;
        winrt::check_hresult(utils->CreateDefaultIncludeHandler(
            includeHandler.put()));

        winrt::com_ptr<IDxcCompiler3> compiler;
        winrt::check_hresult(DxcCreateInstance(
            CLSID_DxcCompiler, IID_PPV_ARGS(compiler.put())));

        winrt::com_ptr<IDxcResult> result;
        winrt::check_hresult(compiler->Compile(
            &dxcBuffer,
            compilerArgs->GetArguments(),
            compilerArgs->GetCount(),
            includeHandler.get(),
            IID_PPV_ARGS(result.put())));

        winrt::com_ptr<IDxcBlob> computeShader;
        winrt::check_hresult(result->GetResult(computeShader.put()));

        std::vector<uint8_t> byteCode(computeShader->GetBufferSize());
        memcpy(byteCode.data(), computeShader->GetBufferPointer(),
            computeShader->GetBufferSize());

        return byteCode;
    }

    winrt::com_ptr<ID3D12RootSignature> CreateRootSignature(
        winrt::com_ptr<ID3D12Device> device,
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters)
    {
        winrt::com_ptr<ID3D12RootSignature> rootSignature;
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
        computeRootSignatureDesc.Init_1_1(
            static_cast<uint32_t>(rootParameters.size()),
            rootParameters.data(), 0, nullptr);

        winrt::com_ptr<ID3DBlob> signature;
        winrt::check_hresult(D3DX12SerializeVersionedRootSignature(
            &computeRootSignatureDesc,
            D3D_ROOT_SIGNATURE_VERSION_1_1,
            signature.put(),
            nullptr));

        winrt::check_hresult(device->CreateRootSignature(
            0,
            signature->GetBufferPointer(),
            signature->GetBufferSize(),
            IID_PPV_ARGS(rootSignature.put())));
        return rootSignature;
    }
};