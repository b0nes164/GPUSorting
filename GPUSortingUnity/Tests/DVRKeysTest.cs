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
    public class DVRKeysTest : TestBase
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
                ref globalHist,
                ref passHist);
            StartCoroutine(KeysFullCoroutine());
        }

        protected override bool KeysTest(
            int testSize,
            bool shouldAscend,
            Type keyType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            m_util.DisableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, true);
            m_dvr.Sort(
                testSize,
                toTest,
                alt,
                globalHist,
                passHist,
                keyType,
                shouldAscend);
            return PostSort(testSize, true, false);
        }

        protected override bool KeysCmdTest(
            int testSize,
            bool shouldAscend,
            Type keyType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            m_util.DisableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, true);
            m_cmd.Clear();
            m_dvr.Sort(
                m_cmd,
                testSize,
                toTest,
                alt,
                globalHist,
                passHist,
                keyType,
                shouldAscend);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostSort(testSize, true, false);
        }

        private IEnumerator KeysFullCoroutine()
        {
            yield return StartCoroutine(KeysTestCoroutine());
            yield return StartCoroutine(KeysCmdTestCoroutine());

            if (m_totalTestPassed == k_totalTestsExpected)
                Debug.Log("ALL DEVICE_RADIX_SORT TESTS PASSED");
            else
                Debug.LogError("DEVICE_RADIX_SORT TESTS FAILED");
        }
    }
}