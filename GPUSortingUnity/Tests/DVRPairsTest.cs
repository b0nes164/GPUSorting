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
    public class DVRPairsTest : TestBase
    {
        [SerializeField]
        ComputeShader deviceRadixSort;

        GPUSorting.Runtime.DeviceRadixSort m_dvr;

        void Start()
        {
            m_sortName = "DVR";
            Initialize();
            m_dvr = new GPUSorting.Runtime.DeviceRadixSort(
                deviceRadixSort,
                k_maxTestSize,
                ref alt,
                ref altPayload,
                ref globalHist,
                ref passHist);
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
            m_dvr.Sort(
                testSize,
                toTest,
                toTestPayload,
                alt,
                altPayload,
                globalHist,
                passHist,
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
            m_dvr.Sort(
                m_cmd,
                testSize,
                toTest,
                toTestPayload,
                alt,
                altPayload,
                globalHist,
                passHist,
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
                Debug.Log("ALL DEVICE_RADIX_SORT PAIRS TESTS PASSED");
            else
                Debug.LogError("DEVICE_RADIX_SORT PAIRS TESTS FAILED");
        }
    }
}