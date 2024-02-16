#pragma once
#include "pch.h"
#include "GPUsorting.h"

class ComputeShader
{
public:
    explicit ComputeShader(
        winrt::com_ptr<ID3D12Device> device, 
        DeviceInfo const& info,
        std::filesystem::path const& shaderPath,
        const wchar_t* entryPoint,
        std::vector<std::wstring> compileArguments,
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters)
    {
        auto byteCode = CompileShader(shaderPath, info, entryPoint,
            compileArguments);
        _rootSig = CreateRootSignature(device, rootParameters);

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
        pipelineDesc.pRootSignature = _rootSig.get();
        pipelineDesc.CS.pShaderBytecode = byteCode.data();
        pipelineDesc.CS.BytecodeLength = byteCode.size();
        winrt::check_hresult(device->CreateComputePipelineState(
            &pipelineDesc, IID_PPV_ARGS(_computePipelineStateDesc.put())));
    }

    void SetPipelineState(winrt::com_ptr<ID3D12GraphicsCommandList> cmdList)
    {
        cmdList->SetPipelineState(_computePipelineStateDesc.get());
        cmdList->SetComputeRootSignature(_rootSig.get());
    }

private:
    winrt::com_ptr<ID3D12RootSignature> _rootSig;
    winrt::com_ptr<ID3D12PipelineState> _computePipelineStateDesc;

    std::vector<uint8_t> CompileShader(
        std::filesystem::path const& shaderPath, 
        DeviceInfo const& info, 
        const wchar_t* entryPoint,
        std::vector<std::wstring> arguments)
    {
        std::vector<uint8_t> byteCode;

        if (info.Supports16BitTypes)
        {
            arguments.push_back(L"-enable-16bit-types");
            arguments.push_back(L"-DENABLE_16_BIT");
        }

        arguments.push_back(L"-O3");
#ifdef _DEBUG
        arguments.push_back(L"-Zi");
#endif

        winrt::com_ptr<IDxcLibrary> library;
        winrt::check_hresult(DxcCreateInstance(
            CLSID_DxcLibrary, IID_PPV_ARGS(library.put())));

        winrt::com_ptr<IDxcCompiler> compiler;
        winrt::check_hresult(DxcCreateInstance(
            CLSID_DxcCompiler, IID_PPV_ARGS(compiler.put())));

        winrt::com_ptr<IDxcIncludeHandler> includeHandler;
        winrt::check_hresult(library->CreateIncludeHandler(
            includeHandler.put()));

        winrt::com_ptr<IDxcBlobEncoding> sourceBlob;
        uint32_t codePage = CP_UTF8;
        winrt::check_hresult(library->CreateBlobFromFile(
            shaderPath.wstring().c_str(), &codePage, sourceBlob.put()));

        std::vector<wchar_t const*> pargs;
        std::transform(arguments.begin(), arguments.end(), 
            back_inserter(pargs), [](auto const& a) { return a.c_str(); });

        winrt::com_ptr<IDxcOperationResult> result;
        HRESULT hr = compiler->Compile(
            sourceBlob.get(),
            shaderPath.wstring().c_str(),
            entryPoint,
            info.SupportedShaderModel.c_str(),
            pargs.data(),
            static_cast<uint32_t>(pargs.size()),
            nullptr,
            0,
            includeHandler.get(),
            result.put());

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
        byteCode.resize(computeShader->GetBufferSize());
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