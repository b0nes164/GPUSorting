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
    public class DeviceRadixSort : GPUSortBase
    {
        private int m_kernelInit = -1;
        private int m_kernelUpsweep = -1;
        private int m_kernelScan = -1;
        private int m_kernelDownsweep = -1;

        private readonly bool k_keysOnly;

        public DeviceRadixSort(
            ComputeShader compute,
            int allocationSize,
            ref ComputeBuffer tempKeyBuffer,
            ref ComputeBuffer tempGlobalHistBuffer,
            ref ComputeBuffer tempPassHistBuffer) : 
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

            tempKeyBuffer = new ComputeBuffer(k_maxKeysAllocated, 4);
            tempGlobalHistBuffer = new ComputeBuffer(k_radix * k_radixPasses, 4);
            tempPassHistBuffer = new ComputeBuffer(k_radix * DivRoundUp(k_maxKeysAllocated, k_partitionSize), 4);
        }

        public DeviceRadixSort(
            ComputeShader compute,
            int allocationSize,
            ref ComputeBuffer tempKeyBuffer,
            ref ComputeBuffer tempPayloadBuffer,
            ref ComputeBuffer tempGlobalHistBuffer,
            ref ComputeBuffer tempPassHistBuffer) :
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

            tempKeyBuffer = new ComputeBuffer(k_maxKeysAllocated, 4);
            tempPayloadBuffer = new ComputeBuffer(k_maxKeysAllocated, 4);
            tempGlobalHistBuffer = new ComputeBuffer(k_radix * k_radixPasses, 4);
            tempPassHistBuffer = new ComputeBuffer(k_radix * DivRoundUp(k_maxKeysAllocated, k_partitionSize), 4);
        }

        private void InitKernels()
        {
            bool isValid;

            if (m_cs)
            {
                m_kernelInit = m_cs.FindKernel("InitDeviceRadixSort");
                m_kernelUpsweep = m_cs.FindKernel("Upsweep");
                m_kernelScan = m_cs.FindKernel("Scan");
                m_kernelDownsweep = m_cs.FindKernel("Downsweep");
            }

            isValid =   m_kernelInit >= 0 &&
                        m_kernelUpsweep >= 0 &&
                        m_kernelScan >= 0 &&
                        m_kernelDownsweep >= 0;

            if (isValid)
            {
                if (!m_cs.IsSupported(m_kernelInit) ||
                    !m_cs.IsSupported(m_kernelUpsweep) ||
                    !m_cs.IsSupported(m_kernelScan) ||
                    !m_cs.IsSupported(m_kernelDownsweep))
                {
                    isValid = false;
                }
            }

            Assert.IsTrue(isValid);
        }

        private void SetStaticRootParameters(
            int numKeys,
            int numThreadBlocks,
            ComputeBuffer _passHistBuffer,
            ComputeBuffer _globalHistBuffer)
        {
            m_cs.SetInt("e_numKeys", numKeys);
            m_cs.SetInt("e_threadBlocks", numThreadBlocks);

            m_cs.SetBuffer(m_kernelInit, "b_globalHist", _globalHistBuffer);

            m_cs.SetBuffer(m_kernelUpsweep, "b_passHist", _passHistBuffer);
            m_cs.SetBuffer(m_kernelUpsweep, "b_globalHist", _globalHistBuffer);

            m_cs.SetBuffer(m_kernelScan, "b_passHist", _passHistBuffer);

            m_cs.SetBuffer(m_kernelDownsweep, "b_passHist", _passHistBuffer);
            m_cs.SetBuffer(m_kernelDownsweep, "b_globalHist", _globalHistBuffer);
        }

        private void SetStaticRootParameters(
            int numKeys,
            int numThreadBlocks,
            CommandBuffer _cmd,
            ComputeBuffer _passHistBuffer,
            ComputeBuffer _globalHistBuffer)
        {
            _cmd.SetComputeIntParam(m_cs, "e_numKeys", numKeys);
            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);

            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_globalHist", _globalHistBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_globalHist", _globalHistBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelScan, "b_passHist", _passHistBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_globalHist", _globalHistBuffer);
        }

        private void Dispatch(
            int numThreadBlocks,
            ComputeBuffer _toSort,
            ComputeBuffer _alt)
        {
            m_cs.Dispatch(m_kernelInit, 1, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                m_cs.SetInt("e_radixShift", radixShift);

                m_cs.SetBuffer(m_kernelUpsweep, "b_sort", _toSort);
                m_cs.Dispatch(m_kernelUpsweep, numThreadBlocks, 1, 1);

                m_cs.Dispatch(m_kernelScan, k_radix, 1, 1);

                m_cs.SetBuffer(m_kernelDownsweep, "b_sort", _toSort);
                m_cs.SetBuffer(m_kernelDownsweep, "b_alt", _alt);
                m_cs.Dispatch(m_kernelDownsweep, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
            }
        }

        private void Dispatch(
            int numThreadBlocks,
            CommandBuffer _cmd,
            ComputeBuffer _toSort,
            ComputeBuffer _alt)
        {
            _cmd.DispatchCompute(m_cs, m_kernelInit, 1, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);

                _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_sort", _toSort);
                _cmd.DispatchCompute(m_cs, m_kernelUpsweep, numThreadBlocks, 1, 1);

                _cmd.DispatchCompute(m_cs, m_kernelScan, k_radix, 1, 1);

                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sort", _toSort);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_alt", _alt);
                _cmd.DispatchCompute(m_cs, m_kernelDownsweep, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
            }
        }

        private void Dispatch(
            int numThreadBlocks,
            ComputeBuffer _toSort,
            ComputeBuffer _toSortPayload,
            ComputeBuffer _alt,
            ComputeBuffer _altPayload)
        {
            m_cs.Dispatch(m_kernelInit, 1, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                m_cs.SetInt("e_radixShift", radixShift);

                m_cs.SetBuffer(m_kernelUpsweep, "b_sort", _toSort);
                m_cs.Dispatch(m_kernelUpsweep, numThreadBlocks, 1, 1);

                m_cs.Dispatch(m_kernelScan, k_radix, 1, 1);

                m_cs.SetBuffer(m_kernelDownsweep, "b_sort", _toSort);
                m_cs.SetBuffer(m_kernelDownsweep, "b_sortPayload", _toSortPayload);
                m_cs.SetBuffer(m_kernelDownsweep, "b_alt", _alt);
                m_cs.SetBuffer(m_kernelDownsweep, "b_altPayload", _altPayload);
                m_cs.Dispatch(m_kernelDownsweep, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
                (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }

        private void Dispatch(
            int numThreadBlocks,
            CommandBuffer _cmd,
            ComputeBuffer _toSort,
            ComputeBuffer _toSortPayload,
            ComputeBuffer _alt,
            ComputeBuffer _altPayload)
        {
            _cmd.DispatchCompute(m_cs, m_kernelInit, 1, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);

                _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_sort", _toSort);
                _cmd.DispatchCompute(m_cs, m_kernelUpsweep, numThreadBlocks, 1, 1);

                _cmd.DispatchCompute(m_cs, m_kernelScan, k_radix, 1, 1);

                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sort", _toSort);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sortPayload", _toSortPayload);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_alt", _alt);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_altPayload", _altPayload);
                _cmd.DispatchCompute(m_cs, m_kernelDownsweep, numThreadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
                (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }

        private void AssertChecksKeys(int _inputSize, System.Type _keyType)
        {
            Assert.IsTrue(k_keysOnly);
            Assert.IsTrue(_inputSize > k_minSize && _inputSize <= k_maxKeysAllocated);
            Assert.IsTrue(
                _keyType == typeof(uint)    ||
                _keyType == typeof(float)   ||
                _keyType == typeof(int));
        }

        private void AssertChecksPairs(int _inputSize, System.Type _keyType, System.Type _payloadType)
        {
            Assert.IsFalse(k_keysOnly);
            Assert.IsTrue(_inputSize > k_minSize && _inputSize <= k_maxKeysAllocated);
            Assert.IsTrue(
                _keyType == typeof(uint)    ||
                _keyType == typeof(float)   ||
                _keyType == typeof(int));
            Assert.IsTrue(
                _payloadType == typeof(uint)    || 
                _payloadType == typeof(float)   || 
                _payloadType == typeof(int));
        }

        //Keys only
        public void Sort(
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            System.Type keyType,
            bool shouldAscend)
        {
            AssertChecksKeys(sortSize, keyType);
            SetKeyTypeKeywords(keyType);
            SetAscendingKeyWords(shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            SetStaticRootParameters(
                sortSize,
                threadBlocks,
                tempPassHistBuffer,
                tempGlobalHistBuffer);
            Dispatch(threadBlocks, toSort, tempKeyBuffer);
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
            System.Type keyType,
            bool shouldAscend)
        {
            AssertChecksKeys(sortSize, keyType);
            SetKeyTypeKeywords(cmd, keyType);
            SetAscendingKeyWords(cmd, shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            SetStaticRootParameters(
                sortSize,
                threadBlocks,
                cmd,
                tempPassHistBuffer,
                tempGlobalHistBuffer);
            Dispatch(threadBlocks, cmd, toSort, tempKeyBuffer);
        }

        //Pairs
        public void Sort(
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempPayloadBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            AssertChecksPairs(sortSize, keyType, payloadType);
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            SetAscendingKeyWords(shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            SetStaticRootParameters(
                sortSize,
                threadBlocks,
                tempPassHistBuffer,
                tempGlobalHistBuffer);
            Dispatch(threadBlocks, toSort, toSortPayload, tempKeyBuffer, tempPayloadBuffer);
        }

        //Pairs
        public void Sort(
            CommandBuffer cmd,
            int sortSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ComputeBuffer tempKeyBuffer,
            ComputeBuffer tempPayloadBuffer,
            ComputeBuffer tempGlobalHistBuffer,
            ComputeBuffer tempPassHistBuffer,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            AssertChecksPairs(sortSize, keyType, payloadType);
            SetKeyTypeKeywords(cmd, keyType);
            SetPayloadTypeKeywords(cmd, payloadType);
            SetAscendingKeyWords(cmd, shouldAscend);
            int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
            SetStaticRootParameters(
                sortSize,
                threadBlocks,
                cmd,
                tempPassHistBuffer,
                tempGlobalHistBuffer);
            Dispatch(threadBlocks, cmd, toSort, toSortPayload, tempKeyBuffer, tempPayloadBuffer);
        }
    }
}