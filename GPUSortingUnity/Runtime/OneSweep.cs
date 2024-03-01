/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 2/28/2024
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
        private readonly int m_kernelInit = -1;
        private readonly int m_kernelGlobalHist = -1;
        private readonly int m_kernelScan = -1;
        private readonly int m_digitBinningPass = -1;

        protected readonly bool m_isValid;
        protected readonly bool m_staticKeysOnly;
        protected readonly int m_maxElements;
        public bool Valid => m_isValid;

        public OneSweep(ComputeShader compute)
        {
            m_cs = compute;
            m_numKeys = 0;
            if (m_cs)
            {
                m_kernelInit = m_cs.FindKernel("InitOneSweep");
                m_kernelGlobalHist = m_cs.FindKernel("GlobalHistogram");
                m_kernelScan = m_cs.FindKernel("Scan");
                m_digitBinningPass = m_cs.FindKernel("DigitBinningPass");
            }

            m_isValid = m_kernelInit >= 0 &&
                        m_kernelGlobalHist >= 0 &&
                        m_kernelScan >= 0 &&
                        m_digitBinningPass >= 0;

            if (m_isValid)
            {
                if (!m_cs.IsSupported(m_kernelInit) ||
                    !m_cs.IsSupported(m_kernelGlobalHist) ||
                    !m_cs.IsSupported(m_kernelScan) ||
                    !m_cs.IsSupported(m_digitBinningPass))
                {
                    m_isValid = false;
                }
            }

            if (m_isValid)
                InitializeKeywords();
        }

        //tempBuffer0 = alt
        //tempBuffer1 = altPayload
        //tempBuffer2 = passHist
        //tempBuffer3 = globalHist
        public OneSweep(
            ComputeShader compute,
            int maxElements,
            ref ComputeBuffer tempBuffer0,
            ref ComputeBuffer tempBuffer2,
            ref ComputeBuffer tempBuffer3,
            ref ComputeBuffer tempBuffer4) : this(compute)
        {
            Assert.IsTrue(
                maxElements >= k_minSize &&
                maxElements <= k_maxSize);
            m_staticMemory = true;
            m_staticKeysOnly = true;
            m_maxElements = maxElements;
            UpdateSizeKeysOnly(maxElements);
            UpdateResources(
                ref tempBuffer0,
                ref tempBuffer2,
                ref tempBuffer3,
                ref tempBuffer4);
        }

        public OneSweep(
            ComputeShader compute,
            int maxElements,
            ref ComputeBuffer tempBuffer0,
            ref ComputeBuffer tempBuffer1,
            ref ComputeBuffer tempBuffer2,
            ref ComputeBuffer tempBuffer3,
            ref ComputeBuffer tempBuffer4) : this(compute)
        {
            Assert.IsTrue(
                maxElements >= k_minSize &&
                maxElements <= k_maxSize);
            m_staticMemory = true;
            m_staticKeysOnly = false;
            m_maxElements = maxElements;
            UpdateSizePairs(maxElements);
            UpdateResources(
                ref tempBuffer0,
                ref tempBuffer1,
                ref tempBuffer2,
                ref tempBuffer3,
                ref tempBuffer4);
        }

        private void UpdateResources(
            ref ComputeBuffer _altBuffer,
            ref ComputeBuffer _passHistBuffer,
            ref ComputeBuffer _globalHistBuffer,
            ref ComputeBuffer _indexBuffer)
        {
            _altBuffer?.Dispose();
            _passHistBuffer?.Dispose();
            _globalHistBuffer?.Dispose();
            _indexBuffer?.Dispose();

            _altBuffer = new ComputeBuffer(m_numKeys, 4);
            _passHistBuffer = new ComputeBuffer(k_radix * k_radixPasses * m_threadBlocks, 4);
            _globalHistBuffer = new ComputeBuffer(k_radix * k_radixPasses, 4);
            _indexBuffer = new ComputeBuffer(k_radixPasses, sizeof(uint));
        }

        private void UpdateResources(
            ref ComputeBuffer _altBuffer,
            ref ComputeBuffer _altPayloadBuffer,
            ref ComputeBuffer _passHistBuffer,
            ref ComputeBuffer _globalHistBuffer,
            ref ComputeBuffer _indexBuffer)
        {
            UpdateResources(
                ref _altBuffer,
                ref _passHistBuffer,
                ref _globalHistBuffer,
                ref _indexBuffer);

            _altPayloadBuffer?.Dispose();
            _altPayloadBuffer = new ComputeBuffer(m_numKeys, 4);
        }

        private void SetStaticRootParameters(
            ComputeBuffer _sortBuffer,
            ComputeBuffer _passHistBuffer,
            ComputeBuffer _globalHistBuffer,
            ComputeBuffer _indexBuffer)
        {
            m_cs.SetInt("e_numKeys", m_numKeys);
            m_cs.SetInt("e_threadBlocks", m_threadBlocks);

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
            CommandBuffer _cmd,
            ComputeBuffer _sortBuffer,
            ComputeBuffer _passHistBuffer,
            ComputeBuffer _globalHistBuffer,
            ComputeBuffer _indexBuffer)
        {
            _cmd.SetComputeIntParam(m_cs, "e_numKeys", m_numKeys);
            _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", m_threadBlocks);

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

        private void Dispatch(ComputeBuffer _toSort, ComputeBuffer _alt)
        {
            m_cs.Dispatch(m_kernelInit, 256, 1, 1);
            m_cs.Dispatch(m_kernelGlobalHist, m_threadBlocks, 1, 1);
            m_cs.Dispatch(m_kernelScan, k_radix, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                m_cs.SetInt("e_radixShift", radixShift);
                m_cs.SetBuffer(m_digitBinningPass, "b_sort", _toSort);
                m_cs.SetBuffer(m_digitBinningPass, "b_alt", _alt);
                m_cs.Dispatch(m_digitBinningPass, m_threadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
            }
        }

        private void Dispatch(
            CommandBuffer _cmd,
            ComputeBuffer _toSort,
            ComputeBuffer _alt)
        {
            _cmd.DispatchCompute(m_cs, m_kernelInit, 256, 1, 1);
            _cmd.DispatchCompute(m_cs, m_kernelGlobalHist, m_threadBlocks, 1, 1);
            _cmd.DispatchCompute(m_cs, m_kernelScan, k_radix, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_sort", _toSort);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_alt", _alt);
                _cmd.DispatchCompute(m_cs, m_digitBinningPass, m_threadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
            }
        }

        private void Dispatch(
            ComputeBuffer _toSort,
            ComputeBuffer _toSortPayload,
            ComputeBuffer _alt,
            ComputeBuffer _altPayload)
        {
            m_cs.Dispatch(m_kernelInit, 256, 1, 1);
            m_cs.Dispatch(m_kernelGlobalHist, m_threadBlocks, 1, 1);
            m_cs.Dispatch(m_kernelScan, k_radix, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                m_cs.SetInt("e_radixShift", radixShift);
                m_cs.SetBuffer(m_digitBinningPass, "b_sort", _toSort);
                m_cs.SetBuffer(m_digitBinningPass, "b_sortPayload", _toSortPayload);
                m_cs.SetBuffer(m_digitBinningPass, "b_alt", _alt);
                m_cs.SetBuffer(m_digitBinningPass, "b_altPayload", _altPayload);
                m_cs.Dispatch(m_digitBinningPass, m_threadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
                (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }

        private void Dispatch(
            CommandBuffer _cmd,
            ComputeBuffer _toSort,
            ComputeBuffer _toSortPayload,
            ComputeBuffer _alt,
            ComputeBuffer _altPayload)
        {
            _cmd.DispatchCompute(m_cs, m_kernelInit, 256, 1, 1);
            _cmd.DispatchCompute(m_cs, m_kernelGlobalHist, m_threadBlocks, 1, 1);
            _cmd.DispatchCompute(m_cs, m_kernelScan, k_radix, 1, 1);

            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_sort", _toSort);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_sortPayload", _toSortPayload);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_alt", _alt);
                _cmd.SetComputeBufferParam(m_cs, m_digitBinningPass, "b_alt", _altPayload);
                _cmd.DispatchCompute(m_cs, m_digitBinningPass, m_threadBlocks, 1, 1);

                (_toSort, _alt) = (_alt, _toSort);
                (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }

        //Resizeable
        //Keys only
        public void Sort(
            int inputSize,
            ComputeBuffer toSort,
            ref ComputeBuffer tempBuffer0,
            ref ComputeBuffer tempBuffer2,
            ref ComputeBuffer tempBuffer3,
            ref ComputeBuffer tempBuffer4,
            System.Type keyType,
            bool shouldAscend)
        {
            Assert.IsTrue(
                Valid &&
                m_staticMemory == false &&
                inputSize >= k_minSize &&
                inputSize <= k_maxSize);
            SetKeyTypeKeywords(keyType);
            SetAscendingKeyWords(shouldAscend);
            m_cs.DisableKeyword(m_sortPairKeyword);
            if (UpdateSizeKeysOnly(inputSize))
            {
                UpdateResources(
                ref tempBuffer0,
                ref tempBuffer2,
                ref tempBuffer3,
                ref tempBuffer4);
            }
            SetStaticRootParameters(
                toSort,
                tempBuffer2,
                tempBuffer3,
                tempBuffer4);
            Dispatch(toSort, tempBuffer0);
        }

        //Static memory allocation
        //Keys only
        public void Sort(
            int inputSize,
            ComputeBuffer toSort,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer2,
            ComputeBuffer tempBuffer3,
            ComputeBuffer tempBuffer4,
            System.Type keyType,
            bool shouldAscend)
        {
            Assert.IsTrue(
                Valid &&
                m_staticMemory == true &&
                m_staticKeysOnly == true &&
                inputSize <= m_maxElements);

            if (inputSize > 0)
            {
                SetKeyTypeKeywords(keyType);
                SetAscendingKeyWords(shouldAscend);
                m_cs.DisableKeyword(m_sortPairKeyword);
                UpdateSizeKeysOnly(inputSize);
                SetStaticRootParameters(
                    toSort,
                    tempBuffer2,
                    tempBuffer3,
                    tempBuffer4);
                Dispatch(toSort, tempBuffer0);
            }
        }

        //Static memory allocation
        //Keys only
        //Command queue
        public void Sort(
            CommandBuffer cmd,
            int inputSize,
            ComputeBuffer toSort,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer2,
            ComputeBuffer tempBuffer3,
            ComputeBuffer tempBuffer4,
            System.Type keyType,
            bool shouldAscend)
        {
            Assert.IsTrue(
                Valid &&
                m_staticMemory == true &&
                m_staticKeysOnly == true &&
                inputSize <= m_maxElements);

            if (inputSize > 0)
            {
                SetKeyTypeKeywords(cmd, keyType);
                SetAscendingKeyWords(cmd, shouldAscend);
                cmd.DisableKeyword(m_cs, m_sortPairKeyword);
                UpdateSizeKeysOnly(inputSize);
                SetStaticRootParameters(
                    cmd,
                    toSort,
                    tempBuffer2,
                    tempBuffer3,
                    tempBuffer4);
                Dispatch(cmd, toSort, tempBuffer0);
            }
        }

        //Resizeable
        //Pairs
        public void Sort(
            int inputSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ref ComputeBuffer tempBuffer0,
            ref ComputeBuffer tempBuffer1,
            ref ComputeBuffer tempBuffer2,
            ref ComputeBuffer tempBuffer3,
            ref ComputeBuffer tempBuffer4,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            Assert.IsTrue(
                Valid &&
                m_staticMemory == false &&
                toSort.count == toSortPayload.count &&
                inputSize >= k_minSize &&
                inputSize <= k_maxSize);

            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            SetAscendingKeyWords(shouldAscend);
            m_cs.EnableKeyword(m_sortPairKeyword);
            if (UpdateSizePairs(inputSize))
            {
                UpdateResources(
                ref tempBuffer0,
                ref tempBuffer1,
                ref tempBuffer2,
                ref tempBuffer3,
                ref tempBuffer4);
            }
            SetStaticRootParameters(
                toSort,
                tempBuffer2,
                tempBuffer3,
                tempBuffer4);
            Dispatch(
                toSort,
                toSortPayload,
                tempBuffer0,
                tempBuffer1);
        }

        //Static memory allocation
        //Pairs
        public void Sort(
            int inputSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer1,
            ComputeBuffer tempBuffer2,
            ComputeBuffer tempBuffer3,
            ComputeBuffer tempBuffer4,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            Assert.IsTrue(
                Valid &&
                m_staticMemory == true &&
                m_staticKeysOnly == false &&
                toSort.count == toSortPayload.count &&
                inputSize <= m_maxElements);

            if (inputSize > 0)
            {
                SetKeyTypeKeywords(keyType);
                SetPayloadTypeKeywords(payloadType);
                SetAscendingKeyWords(shouldAscend);
                m_cs.EnableKeyword(m_sortPairKeyword);
                UpdateSizePairs(inputSize);
                SetStaticRootParameters(
                    toSort,
                    tempBuffer2,
                    tempBuffer3,
                    tempBuffer4);
                Dispatch(
                    toSort,
                    toSortPayload,
                    tempBuffer0,
                    tempBuffer1);
            }
        }

        //Static memory allocation
        //Pairs
        //Command queue
        public void Sort(
            CommandBuffer cmd,
            int inputSize,
            ComputeBuffer toSort,
            ComputeBuffer toSortPayload,
            ComputeBuffer tempBuffer0,
            ComputeBuffer tempBuffer1,
            ComputeBuffer tempBuffer2,
            ComputeBuffer tempBuffer3,
            ComputeBuffer tempBuffer4,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            Assert.IsTrue(
                Valid &&
                m_staticMemory == true &&
                m_staticKeysOnly == false &&
                toSort.count == toSortPayload.count &&
                inputSize <= m_maxElements);

            if (inputSize > 0)
            {
                SetKeyTypeKeywords(cmd, keyType);
                SetPayloadTypeKeywords(cmd, payloadType);
                SetAscendingKeyWords(cmd, shouldAscend);
                cmd.EnableKeyword(m_cs, m_sortPairKeyword);
                UpdateSizeKeysOnly(inputSize);
                SetStaticRootParameters(
                    cmd,
                    toSort,
                    tempBuffer2,
                    tempBuffer3,
                    tempBuffer4);
                Dispatch(
                    cmd,
                    toSort,
                    toSortPayload,
                    tempBuffer0,
                    tempBuffer1);
            }
        }
    }
}