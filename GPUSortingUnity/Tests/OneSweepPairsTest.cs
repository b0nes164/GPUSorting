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
    public class OneSweepPairsTest : TestBase
    {
        [SerializeField]
        ComputeShader oneSweep;

        GPUSorting.Runtime.OneSweep m_os;

        void Start()
        {
            m_sortName = "OneSweep";
            Initialize();
            m_os = new GPUSorting.Runtime.OneSweep(
                oneSweep,
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
            m_os.Sort(
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
            m_os.Sort(
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
                Debug.Log("ALL ONE_SWEEP PAIRS TESTS PASSED");
            else
                Debug.LogError("ONE_SWEEP PAIRS TESTS FAILED");
        }
    }
}