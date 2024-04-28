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
    public class ForwardSweep : OneSweep
    {
        public ForwardSweep(
            ComputeShader compute,
            int allocationSize,
            ref ComputeBuffer tempKeyBuffer,
            ref ComputeBuffer tempGlobalHistBuffer,
            ref ComputeBuffer tempPassHistBuffer,
            ref ComputeBuffer tempIndexBuffer) :
            base(
                compute,
                allocationSize,
                ref tempKeyBuffer,
                ref tempGlobalHistBuffer,
                ref tempPassHistBuffer,
                ref tempIndexBuffer)
        {
        }

        public ForwardSweep(
            ComputeShader compute,
            int allocationSize,
            ref ComputeBuffer tempKeyBuffer,
            ref ComputeBuffer tempPayloadBuffer,
            ref ComputeBuffer tempGlobalHistBuffer,
            ref ComputeBuffer tempPassHistBuffer,
            ref ComputeBuffer tempIndexBuffer) :
            base(
                compute,
                allocationSize,
                ref tempKeyBuffer,
                ref tempPayloadBuffer,
                ref tempGlobalHistBuffer,
                ref tempPassHistBuffer,
                ref tempIndexBuffer)
        {
        }

        protected override void InitKernels()
        {
            bool isValid;
            if (m_cs)
            {
                m_kernelInit = m_cs.FindKernel("InitSweep");
                m_kernelGlobalHist = m_cs.FindKernel("GlobalHistogram");
                m_kernelScan = m_cs.FindKernel("Scan");
                m_digitBinningPass = m_cs.FindKernel("ForwardSweep");
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
    }
}

