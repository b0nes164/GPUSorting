/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/28/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Assertions;

namespace GPUSorting.Runtime
{
    public class OneSweep : GPUSortBase
    {
        protected const int k_globalHistPartSize = 32768;

        protected int m_kernelInit = -1;
        protected int m_kernelGlobalHist = -1;
        protected int m_kernelScan = -1;
        protected int m_digitBinningPass = -1;

        protected readonly bool k_keysOnly;

        //keys
        public OneSweep(
            ComputeShader compute,
            int allocationSize,
            ref ComputeBuffer tempKeyBuffer,
            ref ComputeBuffer tempGlobalHistBuffer,
            ref ComputeBuffer tempPassHistBuffer,
            ref ComputeBuffer tempIndexBuffer) :
            base(
                compute,
                allocationSize)
        {
            InitKernels();
            m_cs.DisableKeyword(m_sortPairKeyword);
            k_keysOnly = true;

            tempKeyBuffer?.Dispose();
            tempGlobalHistBuffer?.Dispose();
            tempPassHistBuffer?.Dispose();
            tempIndexBuffer?.Dispose();

            tempKeyBuffer = new ComputeBuffer(k_maxKeysAllocated, 4);
            tempGlobalHistBuffer = new ComputeBuffer(k_radix * k_radixPasses, 4);
            tempPassHistBuffer = new ComputeBuffer(k_radix * DivRoundUp(k_maxKeysAllocated, k_partitionSize) * k_radixPasses, 4);
            tempIndexBuffer = new ComputeBuffer(k_radixPasses, sizeof(uint));
        }

        //pairs
        public OneSweep(
            ComputeShader compute,
            int allocationSize,
            ref ComputeBuffer tempKeyBuffer,
            ref ComputeBuffer tempPayloadBuffer,
            ref ComputeBuffer tempGlobalHistBuffer,
            ref ComputeBuffer tempPassHistBuffer,
            ref ComputeBuffer tempIndexBuffer) :
            base(
                compute,
                allocationSize)
        {
            InitKernels();
            m_cs.EnableKeyword(m_sortPairKeyword);
            k_keysOnly = false;

            tempKeyBuffer?.Dispose();
            tempPayloadBuffer?.Dispose();
            tempGlobalHistBuffer?.Dispose();
            tempPassHistBuffer?.Dispose();
            tempIndexBuffer?.Dispose();

            tempKeyBuffer = new ComputeBuffer(k_maxKeysAllocated, 4);
            tempPayloadBuffer = new ComputeBuffer(k_maxKeysAllocated, 4);
            tempGlobalHistBuffer = new ComputeBuffer(k_radix * k_radixPasses, 4);
            tempPassHistBuffer = new ComputeBuffer(k_radix * DivRoundUp(k_maxKeysAllocated, k_partitionSize) * k_radixPasses, 4);
            tempIndexBuffer = new ComputeBuffer(k_radixPasses, sizeof(uint));
        }

        protected virtual void InitKernels()
        {
            bool isValid;
            if (m_cs)
            {
                m_kernelInit = m_cs.FindKernel("InitSweep");
                m_kernelGlobalHist = m_cs.FindKernel("GlobalHistogram");
                m_kernelScan = m_cs.FindKernel("Scan");
                m_digitBinningPass = m_cs.FindKernel("DigitBinningPass");
            }

            isValid = m_kernelInit >= 0 &&
                        m_kernelGlobalHist >= 0 &&
                        m_kernelScan >= 0 &&
                        m_digitBinningPass >= 0;

            if (isValid)
            {
                if (!m_cs.IsSupported(m_kernelInit) ||
                    !m_cs.IsSupported(m_kernelGlobalHist) ||
                    !m_cs.IsSupported(m_kernelScan) ||
                    !m_cs.IsSupported(m_digitBinningPass))
                {
                    isValid = false;
                }
            }

            Assert.IsTrue(isValid);
        }

        private void SetStaticRootParameters(
            int numKeys,
            ComputeBuffer _sortBuffer,
            ComputeBuffer _passHistBuffer,
            ComputeBuffer _globalHistBuffer,
            ComputeBuffer _indexBuffer)
        {
            m_cs.SetInt("e_numKeys", numKeys);

            m_cs.SetBuffer(m_kernelInit, "b_passHist", _passHistBuffer);
            m_cs.SetBuffer(m_kernelInit, "b_globalHist", _globalHistBuffer);
            m_cs.SetBuffer(m_kernelInit, "b_index", _indexBuffer);

            m_cs.SetBuffer(m_kernelGlobalHist, "b_sort", _sortBuffer);
            m_cs.SetBuffer(m_kernelGlobalHist, "b_globalHist", _globalHistBuffer);

            m_cs.SetBuffer(m_kernelScan, "b_passHist", _passHistBuffer);
            m_cs.SetBuffer(m_kernelScan, "b_globalHist", _globalHistBuffer);

            m_cs.SetBuffer(m_digitBinningPass, "b_passHist", _passHistBuffer);
            m_cs.SetBuffer(m_digitBinningPass, "b_index", _indexBuffer);
        }

        private void SetStaticRootParameters(
            int numKeys,
            CommandBuffer _cmd,
            ComputeBuffer _sortBuffer,
            ComputeBuffer _passHistBuffer,
            ComputeBuffer _globalHistBuffer,
            ComputeBuffer _indexBuffer)
        {
            _cmd.SetComputeIntParam(m_cs, "e_numKeys", numKeys);

            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_globalHist", _globalHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_index", _indexBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelGlobalHist, "b_sort", _sortBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelGlobalHist, "b_globalHist", _globalHistBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelScan, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelScan, "b_globalHist", _globalHistBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_index", _indexBuffer);
        }

        private void Dispatch(
            int numThreadBlocks,
            int globalHistThreadBlocks,
            ComputeBuffer _toSort,
            ComputeBuffer _alt)
        {
            m_cs.SetInt("e_threadBlocks", numThreadBlocks);
            m_cs.Dispatch(m_kernelInit, 256, 1, 1);

            m_cs.SetInt("e_threadBlocks", globalHistThreadBlocks);
            m_cs.Dispatch(m_kernelGlobalHist, globalHistThreadBlocks, 1, 1);

            m_cs.SetInt("e_threadBlocks", numThreadBlocks);
            m_cs.Dispatch(m_kernelScan, k_radixPasses, 1, 1);
            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                m_cs.SetInt("e_radixShift", radixShift);
                m_cs.SetBuffer(m_digitBinningPass, "b_sort", _toSort);
                m_cs.SetBuffer(m_digitBinningPass, "b_alt", _alt);
                m_cs.Dispatch(m_digitBinningPass, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
            }
        }

        private void Dispatch(
            int numThreadBlocks,
            int globalHistThreadBlocks,
            CommandBuffer _cmd,
            ComputeBuffer _toSort,
            ComputeBuffer _alt)
        {
            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);
            _cmd.DispatchCompute(m_cs, m_kernelInit, 256, 1, 1);

            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", globalHistThreadBlocks);
            _cmd.DispatchCompute(m_cs, m_kernelGlobalHist, globalHistThreadBlocks, 1, 1);

            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);
            _cmd.DispatchCompute(m_cs, m_kernelScan, k_radixPasses, 1, 1);
            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_sort", _toSort);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_alt", _alt);
                _cmd.DispatchCompute(m_cs, m_digitBinningPass, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
            }
        }

        private void Dispatch(
            int numThreadBlocks,
            int globalHistThreadBlocks,
            ComputeBuffer _toSort,
            ComputeBuffer _toSortPayload,
            ComputeBuffer _alt,
            ComputeBuffer _altPayload)
        {
            m_cs.SetInt("e_threadBlocks", numThreadBlocks);
            m_cs.Dispatch(m_kernelInit, 256, 1, 1);

            m_cs.SetInt("e_threadBlocks", globalHistThreadBlocks);
            m_cs.Dispatch(m_kernelGlobalHist, globalHistThreadBlocks, 1, 1);

            m_cs.SetInt("e_threadBlocks", numThreadBlocks);
            m_cs.Dispatch(m_kernelScan, k_radixPasses, 1, 1);
            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                m_cs.SetInt("e_radixShift", radixShift);
                m_cs.SetBuffer(m_digitBinningPass, "b_sort", _toSort);
                m_cs.SetBuffer(m_digitBinningPass, "b_sortPayload", _toSortPayload);
                m_cs.SetBuffer(m_digitBinningPass, "b_alt", _alt);
                m_cs.SetBuffer(m_digitBinningPass, "b_altPayload", _altPayload);
                m_cs.Dispatch(m_digitBinningPass, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
                (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }

        private void Dispatch(
            int numThreadBlocks,
            int globalHistThreadBlocks,
            CommandBuffer _cmd,
            ComputeBuffer _toSort,
            ComputeBuffer _toSortPayload,
            ComputeBuffer _alt,
            ComputeBuffer _altPayload)
        {
            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);
            _cmd.DispatchCompute(m_cs, m_kernelInit, 256, 1, 1);

            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", globalHistThreadBlocks);
            _cmd.DispatchCompute(m_cs, m_kernelGlobalHist, globalHistThreadBlocks, 1, 1);

            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);
            _cmd.DispatchCompute(m_cs, m_kernelScan, k_radixPasses, 1, 1);
            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_sort", _toSort);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_sortPayload", _toSortPayload);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_alt", _alt);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_altPayload", _altPayload);
                _cmd.DispatchCompute(m_cs, m_digitBinningPass, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
                (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }

        private void AssertChecksKeys(int _inputSize, System.Type _keyType)
        {
            Assert.IsTrue(k_keysOnly);
            Assert.IsTrue(_inputSize > k_minSize && _inputSize <= k_maxKeysAllocated);
            Assert.IsTrue(
                _keyType == typeof(uint) ||
                _keyType == typeof(float) ||
                _keyType == typeof(int));
        }

        private void AssertChecksPairs(int _inputSize, System.Type _keyType, System.Type _payloadType)
        {
            Assert.IsFalse(k_keysOnly);
            Assert.IsTrue(_inputSize > k_minSize && _inputSize <= k_maxKeysAllocated);
            Assert.IsTrue(
                _keyType == typeof(uint) ||
                _keyType == typeof(float) ||
                _keyType == typeof(int));
            Assert.IsTrue(
                _payloadType == typeof(uint) ||
                _payloadType == typeof(float) ||
                _payloadType == typeof(int));
        }

        //Keys only
        public void Sort(
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            ComputeBuffer tempIndexBuffer,
            System.Type keyType,
            bool shouldAscend)
        {
            AssertChecksKeys(sortSize, keyType);
            SetKeyTypeKeywords(keyType);
            SetAscendingKeyWords(shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            int globalHistThreadBlocks = DivRoundUp(sortSize, k_globalHistPartSize);
            SetStaticRootParameters(
                sortSize,
                toSort,
                tempPassHistBuffer,
                tempGlobalHistBuffer,
                tempIndexBuffer);
            Dispatch(
                threadBlocks,
                globalHistThreadBlocks,
                toSort,
                tempKeyBuffer);
        }

        //Keys only
        //Command queue
        public void Sort(
            CommandBuffer cmd,
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            ComputeBuffer tempIndexBuffer,
            System.Type keyType,
            bool shouldAscend)
        {
            AssertChecksKeys(sortSize, keyType);
            SetKeyTypeKeywords(cmd, keyType);
            SetAscendingKeyWords(cmd, shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            int globalHistThreadBlocks = DivRoundUp(sortSize, k_globalHistPartSize);
            SetStaticRootParameters(
                sortSize,
                cmd,
                toSort,
                tempPassHistBuffer,
                tempGlobalHistBuffer,
                tempIndexBuffer);
            Dispatch(
                threadBlocks,
                globalHistThreadBlocks,
                cmd,
                toSort,
                tempKeyBuffer);
        }

        public void Sort(
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempPayloadBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            ComputeBuffer tempIndexBuffer,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            AssertChecksPairs(sortSize, keyType, payloadType);
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            SetAscendingKeyWords(shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            int globalHistThreadBlocks = DivRoundUp(sortSize, k_globalHistPartSize);
            SetStaticRootParameters(
                sortSize,
                toSort,
                tempPassHistBuffer,
                tempGlobalHistBuffer,
                tempIndexBuffer);
            Dispatch(
                threadBlocks,
                globalHistThreadBlocks,
                toSort,
                toSortPayload,
                tempKeyBuffer,
                tempPayloadBuffer);
        }

        public void Sort(
            CommandBuffer cmd,
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempPayloadBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            ComputeBuffer tempIndexBuffer,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            AssertChecksPairs(sortSize, keyType, payloadType);
            SetKeyTypeKeywords(cmd, keyType);
            SetPayloadTypeKeywords(cmd, payloadType);
            SetAscendingKeyWords(cmd, shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            int globalHistThreadBlocks = DivRoundUp(sortSize, k_globalHistPartSize);
            SetStaticRootParameters(
                sortSize,
                cmd,
                toSort,
                tempPassHistBuffer,
                tempGlobalHistBuffer,
                tempIndexBuffer);
            Dispatch(
                threadBlocks,
                globalHistThreadBlocks,
                cmd,
                toSort,
                toSortPayload,
                tempKeyBuffer,
                tempPayloadBuffer);
        }
    }
}