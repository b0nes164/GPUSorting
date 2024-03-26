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

class ComputeKernelBase
{
    winrt::com_ptr<ID3D12RootSignature> m_rootSignature;
    winrt::com_ptr<ID3D12PipelineState> m_computePipelineStateDesc;

public:
    ComputeKernelBase(
        winrt::com_ptr<ID3D12Device> device, 
        const GPUSorting::DeviceInfo& info,
        const std::filesystem::path& shaderPath,
        const wchar_t* entryPoint,
        const std::vector<std::wstring>& compileArguments,
        const std::vector<CD3DX12_ROOT_PARAMETER1>& rootParams)
    {
        auto byteCode = CompileShader(
            shaderPath,
            info,
            entryPoint,
            compileArguments);

        CreateRootSignature(
            device, 
            rootParams);

        CreatePipelineStateDesc(
            device,
            byteCode);
    }

protected:
    const uint32_t k_isNotPartialBitFlag = 0;
    const uint32_t k_isPartialBitFlag = 1;
    const uint32_t k_maxDim = 65535;

    //Slightly scuffed, as we cannot forward declare and call the function
    //in the base constructor without breaking things
    virtual const std::vector<CD3DX12_ROOT_PARAMETER1> CreateRootParameters() = 0;

    void SetPipelineState(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList)
    {
        cmdList->SetPipelineState(m_computePipelineStateDesc.get());
        cmdList->SetComputeRootSignature(m_rootSignature.get());
    }

private:
    std::vector<uint8_t> CompileShader(
        const std::filesystem::path& shaderPath, 
        const GPUSorting::DeviceInfo& info,
        const wchar_t* entryPoint,
        const std::vector<std::wstring>& arguments)
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
        HRESULT hr = compiler->Compile(
            &dxcBuffer,
            compilerArgs->GetArguments(),
            compilerArgs->GetCount(),
            includeHandler.get(),
            IID_PPV_ARGS(result.put()));

        if (SUCCEEDED(hr))
        {
            winrt::check_hresult(result->GetStatus(&hr));
        }

        if (FAILED(hr))
        {
            if (result)
            {
                winrt::com_ptr<IDxcBlobEncoding> errorsBlob;
                HRESULT getErrorBufferResult =
                    result->GetErrorBuffer(errorsBlob.put());
                if (SUCCEEDED(getErrorBufferResult))
                {
                    std::cout << "Details: ";
                    std::cout << static_cast<const char*>(
                        errorsBlob->GetBufferPointer());
                    std::cout << "\n\n";
                }
            }
            winrt::check_hresult(hr);
        }

        winrt::com_ptr<IDxcBlob> computeShader;
        winrt::check_hresult(result->GetResult(computeShader.put()));
        std::vector<uint8_t> byteCode(computeShader->GetBufferSize());
        memcpy(byteCode.data(), computeShader->GetBufferPointer(),
            computeShader->GetBufferSize());

        return byteCode;
    }

    void CreateRootSignature(
        winrt::com_ptr<ID3D12Device> device,
        const std::vector<CD3DX12_ROOT_PARAMETER1>& rootParams)
    {
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
        computeRootSignatureDesc.Init_1_1(
            static_cast<uint32_t>(rootParams.size()),
            rootParams.data(), 0, nullptr);

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
            IID_PPV_ARGS(m_rootSignature.put())));
    }

    void CreatePipelineStateDesc(
        winrt::com_ptr<ID3D12Device> device,
        const std::vector<uint8_t>& byteCode)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
        pipelineDesc.pRootSignature = m_rootSignature.get();
        pipelineDesc.CS.pShaderBytecode = byteCode.data();
        pipelineDesc.CS.BytecodeLength = byteCode.size();
        winrt::check_hresult(device->CreateComputePipelineState(
            &pipelineDesc, IID_PPV_ARGS(m_computePipelineStateDesc.put())));
    }
};