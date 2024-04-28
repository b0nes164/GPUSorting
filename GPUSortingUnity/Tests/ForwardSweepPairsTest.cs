/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/28/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
using System;
using System.Collections;
using UnityEngine;

namespace GPUSorting.Tests
{
    public class ForwardSweepPairsTest : TestBase
    {
        [SerializeField]
        ComputeShader forwardSweep;

        GPUSorting.Runtime.ForwardSweep m_fs;

        void Start()
        {
            m_sortName = "ForwardSweep";
            Initialize();
            m_fs = new GPUSorting.Runtime.ForwardSweep(
                forwardSweep,
                k_maxTestSize,
                ref alt,
                ref altPayload,
                ref globalHist,
                ref passHist,
                ref index);
            StartCoroutine(PairsFullCoroutine());
        }

        protected override bool PairsTest(
            int testSize,
            bool shouldAscend,
            Type keyType,
            Type payloadType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            m_util.EnableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, false);
            m_fs.Sort(
                testSize,
                toTest,
                toTestPayload,
                alt,
                altPayload,
                globalHist,
                passHist,
                index,
                keyType,
                payloadType,
                shouldAscend);
            return PostSort(testSize, false, false);
        }

        protected override bool PairsCmdTest(
            int testSize,
            bool shouldAscend,
            Type keyType,
            Type payloadType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            m_util.EnableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, false);
            m_cmd.Clear();
            m_fs.Sort(
                m_cmd,
                testSize,
                toTest,
                toTestPayload,
                alt,
                altPayload,
                globalHist,
                passHist,
                index,
                keyType,
                payloadType,
                shouldAscend);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostSort(testSize, false, false);
        }

        private IEnumerator PairsFullCoroutine()
        {
            yield return StartCoroutine(PairsTestCoroutine());
            yield return StartCoroutine(PairsCmdTestCoroutine());

            if (m_totalTestPassed == k_totalTestsExpected)
                Debug.Log("ALL FORWARD_SWEEP PAIRS TESTS PASSED");
            else
                Debug.LogError("FORWARD_SWEEP PAIRS TESTS FAILED");
        }
    }
}