/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/28/2024
 * https://github.com/b0nes164/GPUSorting
 * 
 ******************************************************************************/
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using GPUSorting.Runtime;

namespace GPUSorting.Runtime
{
    public class Tests : MonoBehaviour
    {
        [SerializeField]
        ComputeShader dvr;

        [SerializeField]
        ComputeShader oneSweep;

        [SerializeField]
        ComputeShader m_util;

        private ComputeBuffer toTest;
        private ComputeBuffer toTestPayload;
        private ComputeBuffer alt;
        private ComputeBuffer altPayload;
        private ComputeBuffer globalHist;
        private ComputeBuffer passHist;
        private ComputeBuffer index;
        private ComputeBuffer errCount;

        private GPUSorting.Runtime.DeviceRadixSort m_dvr;
        private GPUSorting.Runtime.OneSweep m_oneSweep;
        private CommandBuffer m_cmd;

        private LocalKeyword m_keyIntKeyword;
        private LocalKeyword m_keyUintKeyword;
        private LocalKeyword m_keyFloatKeyword;
        private LocalKeyword m_payloadIntKeyword;
        private LocalKeyword m_payloadUintKeyword;
        private LocalKeyword m_payloadFloatKeyword;
        private LocalKeyword m_ascendKeyword;
        private LocalKeyword m_sortPairKeyword;

        private const int k_validatePartSize = 2048;
        private const int k_pairsPartitionSize = 7680;
        private const int k_maxTestSize = 1 << 21;

        private bool m_isValid;
        private int m_kernelInitRandom = -1;
        private int m_kernelClearErrors = -1;
        private int m_kernelValidate = -1;

        private int m_dvrResizePassed = 0;
        private int m_dvrKeysStaticPassed = 0;
        private int m_dvrKeysCmdPassed = 0;
        private int m_dvrPairsStaticPassed = 0;
        private int m_dvrPairsCmdPassed = 0;
        private int m_dvrTotalTestsPassed = 0;

        private int m_osResizePassed = 0;
        private int m_osKeysStaticPassed = 0;
        private int m_osKeysCmdPassed = 0;
        private int m_osPairsStaticPassed = 0;
        private int m_osPairsCmdPassed = 0;
        private int m_osTotalTestsPassed = 0;

        private const int k_resizeTestsExpected = 96;
        private const int k_staticKeysOnlyTestsExpected = 96;
        private const int k_cmdKeysTestsExpected = 96;
        private const int k_staticPairsTestsExpected = 96;
        private const int k_cmdPairsTestsExpected = k_pairsPartitionSize * 6;
        private const int k_totalTestsExpected =
            k_resizeTestsExpected +
            k_staticKeysOnlyTestsExpected +
            k_cmdKeysTestsExpected +
            k_staticPairsTestsExpected +
            k_cmdPairsTestsExpected;

        private void Start()
        {
            if (m_util)
            {
                m_kernelInitRandom = m_util.FindKernel("InitSortInput");
                m_kernelClearErrors = m_util.FindKernel("ClearErrorCount");
                m_kernelValidate = m_util.FindKernel("Validate");
            }

            m_isValid = m_kernelInitRandom >= 0 &&
                          m_kernelClearErrors >= 0 &&
                          m_kernelValidate >= 0;

            if (m_isValid)
            {
                if (!m_util.IsSupported(m_kernelInitRandom) ||
                    !m_util.IsSupported(m_kernelClearErrors) ||
                    !m_util.IsSupported(m_kernelValidate))
                {
                    m_isValid = false;
                }
            }

            if (m_isValid)
            {
                m_keyUintKeyword = new LocalKeyword(m_util, "KEY_UINT");
                m_keyIntKeyword = new LocalKeyword(m_util, "KEY_INT");
                m_keyFloatKeyword = new LocalKeyword(m_util, "KEY_FLOAT");
                m_payloadUintKeyword = new LocalKeyword(m_util, "PAYLOAD_UINT");
                m_payloadIntKeyword = new LocalKeyword(m_util, "PAYLOAD_INT");
                m_payloadFloatKeyword = new LocalKeyword(m_util, "PAYLOAD_FLOAT");
                m_ascendKeyword = new LocalKeyword(m_util, "SHOULD_ASCEND");
                m_sortPairKeyword = new LocalKeyword(m_util, "SORT_PAIRS");
                m_cmd = new CommandBuffer();
                InitializeTestResources();
                Debug.Log("Test initialization success.");
            }

            StartCoroutine(TestAll());
        }

        private static int DivRoundUp(int x, int y)
        {
            return (x + y - 1) / y;
        }

        private void InitializeTestResources()
        {
            toTest = new ComputeBuffer(k_maxTestSize, sizeof(uint));
            toTestPayload = new ComputeBuffer(k_maxTestSize, sizeof(uint));
            errCount = new ComputeBuffer(1, sizeof(uint));
        }

        private void SetKeyTypeKeywords(System.Type _type)
        {
            if (_type == typeof(int))
            {
                m_util.EnableKeyword(m_keyIntKeyword);
                m_util.DisableKeyword(m_keyUintKeyword);
                m_util.DisableKeyword(m_keyFloatKeyword);
            }

            if (_type == typeof(uint))
            {
                m_util.DisableKeyword(m_keyIntKeyword);
                m_util.EnableKeyword(m_keyUintKeyword);
                m_util.DisableKeyword(m_keyFloatKeyword);
            }

            if (_type == typeof(float))
            {
                m_util.DisableKeyword(m_keyIntKeyword);
                m_util.DisableKeyword(m_keyUintKeyword);
                m_util.EnableKeyword(m_keyFloatKeyword);
            }
        }

        private void SetPayloadTypeKeywords(System.Type _type)
        {
            if (_type == typeof(int))
            {
                m_util.EnableKeyword(m_payloadIntKeyword);
                m_util.DisableKeyword(m_payloadUintKeyword);
                m_util.DisableKeyword(m_payloadFloatKeyword);
            }

            if (_type == typeof(uint))
            {
                m_util.DisableKeyword(m_payloadIntKeyword);
                m_util.EnableKeyword(m_payloadUintKeyword);
                m_util.DisableKeyword(m_payloadFloatKeyword);
            }

            if (_type == typeof(float))
            {
                m_util.DisableKeyword(m_payloadIntKeyword);
                m_util.DisableKeyword(m_payloadUintKeyword);
                m_util.EnableKeyword(m_payloadFloatKeyword);
            }
        }

        private void SetAscendingKeyWords(bool _shouldAscend)
        {
            if (_shouldAscend)
                m_util.EnableKeyword(m_ascendKeyword);
            else
                m_util.DisableKeyword(m_ascendKeyword);
        }

        private void PreSort(int _testSize, int _seed, bool keysOnly)
        {
            m_util.SetInt("e_numKeys", _testSize);
            m_util.SetInt("e_seed", _seed);

            m_util.SetBuffer(m_kernelInitRandom, "b_sort", toTest);
            if (keysOnly == false)
                m_util.SetBuffer(m_kernelInitRandom, "b_sortPayload", toTestPayload);
            m_util.Dispatch(m_kernelInitRandom, 256, 1, 1);
        }

        private bool PostSort(int _testSize, bool keysOnly, bool shouldPrint)
        {
            m_util.SetBuffer(m_kernelClearErrors, "b_errorCount", errCount);
            m_util.Dispatch(m_kernelClearErrors, 1, 1, 1);

            m_util.SetInt("e_threadBlocks", DivRoundUp(_testSize, k_validatePartSize));
            m_util.SetBuffer(m_kernelValidate, "b_sort", toTest);
            if (keysOnly == false)
                m_util.SetBuffer(m_kernelValidate, "b_sortPayload", toTestPayload);
            m_util.SetBuffer(m_kernelValidate, "b_errorCount", errCount);
            m_util.Dispatch(m_kernelValidate, DivRoundUp(_testSize, k_validatePartSize), 1, 1);
            uint[] errors = new uint[1];
            errCount.GetData(errors);

            if (errors[0] == 0)
            {
                if (shouldPrint)
                    Debug.Log("Test passed");
                return true;
            }
            else
            {
                if (shouldPrint)
                    Debug.LogError("Test Failed: " + errors[0] + " errors.");
                return false;
            }
        }

        private bool DVRKeys(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            m_util.DisableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, true);
            m_dvr.Sort(
                testSize,
                toTest,
                ref alt,
                ref passHist,
                ref globalHist,
                keyType,
                shouldAscend);
            return PostSort(testSize, true, false);
        }

        private bool DVRKeysStatic(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
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
                passHist,
                globalHist,
                keyType,
                shouldAscend);
            return PostSort(testSize, true, false);
        }

        private bool DVRKeysCmd(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
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
                passHist,
                globalHist,
                keyType,
                shouldAscend);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostSort(testSize, true, false);
        }

        private bool DVRPairs(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
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
                ref alt,
                ref altPayload,
                ref passHist,
                ref globalHist,
                keyType,
                payloadType,
                shouldAscend);
            return PostSort(testSize, false, false);
        }

        private bool DVRPairsStatic(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
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
                passHist,
                globalHist,
                keyType,
                payloadType,
                shouldAscend);
            return PostSort(testSize, false, false);
        }

        private bool DVRPairsCmd(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
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
                passHist,
                globalHist,
                keyType,
                payloadType,
                shouldAscend);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostSort(testSize, false, false);
        }

        private bool OSKeys(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            m_util.DisableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, true);
            m_oneSweep.Sort(
                testSize,
                toTest,
                ref alt,
                ref passHist,
                ref globalHist,
                ref index,
                keyType,
                shouldAscend);
            return PostSort(testSize, true, false);
        }

        private bool OSKeysStatic(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            m_util.DisableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, true);
            m_oneSweep.Sort(
                testSize,
                toTest,
                alt,
                passHist,
                globalHist,
                index,
                keyType,
                shouldAscend);
            return PostSort(testSize, true, false);
        }

        private bool OSKeysCmd(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            m_util.DisableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, true);
            m_cmd.Clear();
            m_oneSweep.Sort(
                m_cmd,
                testSize,
                toTest,
                alt,
                passHist,
                globalHist,
                index,
                keyType,
                shouldAscend);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostSort(testSize, true, false);
        }

        private bool OSPairs(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            m_util.EnableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, false);
            m_oneSweep.Sort(
                testSize,
                toTest,
                toTestPayload,
                ref alt,
                ref altPayload,
                ref passHist,
                ref globalHist,
                ref index,
                keyType,
                payloadType,
                shouldAscend);
            return PostSort(testSize, false, false);
        }

        private bool OSPairsStatic(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            m_util.EnableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, false);
            m_oneSweep.Sort(
                testSize,
                toTest,
                toTestPayload,
                alt,
                altPayload,
                passHist,
                globalHist,
                index,
                keyType,
                payloadType,
                shouldAscend);
            return PostSort(testSize, false, false);
        }

        private bool OSPairsCmd(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
            int seed)
        {
            SetKeyTypeKeywords(keyType);
            SetPayloadTypeKeywords(payloadType);
            m_util.EnableKeyword(m_sortPairKeyword);
            SetAscendingKeyWords(shouldAscend);
            PreSort(testSize, seed, false);
            m_cmd.Clear();
            m_oneSweep.Sort(
                testSize,
                toTest,
                toTestPayload,
                alt,
                altPayload,
                passHist,
                globalHist,
                index,
                keyType,
                payloadType,
                shouldAscend);
            Graphics.ExecuteCommandBuffer(m_cmd);
            return PostSort(testSize, false, false);
        }

        private IEnumerator DVRResizeTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_dvrResizePassed +=
                    DVRKeys(i, true, typeof(uint), i) ? 1 : 0;
                yield return m_dvrResizePassed +=
                    DVRKeys(i, true, typeof(int), i) ? 1 : 0;
                yield return m_dvrResizePassed +=
                    DVRKeys(i, true, typeof(float), i) ? 1 : 0;
            }

            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_dvrResizePassed +=
                    DVRPairs(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_dvrResizePassed +=
                    DVRPairs(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_dvrResizePassed +=
                    DVRPairs(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
            }

            m_dvrTotalTestsPassed += m_dvrResizePassed;
            if (m_dvrResizePassed == k_resizeTestsExpected)
                Debug.Log("96 / 96 All DVR resize tests passed.");
            else
                Debug.LogError(m_dvrResizePassed + " / 96 DVR resize test failed.");
        }

        private IEnumerator DVRKeysStaticTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_dvrKeysStaticPassed +=
                    DVRKeysStatic(i, true, typeof(uint), i) ? 1 : 0;
                yield return m_dvrKeysStaticPassed +=
                    DVRKeysStatic(i, false, typeof(uint), i) ? 1 : 0;
                yield return m_dvrKeysStaticPassed +=
                    DVRKeysStatic(i, true, typeof(int), i) ? 1 : 0;
                yield return m_dvrKeysStaticPassed +=
                    DVRKeysStatic(i, false, typeof(int), i) ? 1 : 0;
                yield return m_dvrKeysStaticPassed +=
                    DVRKeysStatic(i, true, typeof(float), i) ? 1 : 0;
                yield return m_dvrKeysStaticPassed +=
                    DVRKeysStatic(i, false, typeof(float), i) ? 1 : 0;
            }

            m_dvrTotalTestsPassed += m_dvrKeysStaticPassed;
            if (m_dvrKeysStaticPassed == k_staticKeysOnlyTestsExpected)
                Debug.Log("96 / 96 DVR static keys tests passed.");
            else
                Debug.LogError(m_dvrKeysStaticPassed + " / 96 DVR static keys tests failed.");
        }

        private IEnumerator DVRKeysCmdTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_dvrKeysCmdPassed +=
                    DVRKeysCmd(i, true, typeof(uint), i) ? 1 : 0;
                yield return m_dvrKeysCmdPassed +=
                    DVRKeysCmd(i, false, typeof(uint), i) ? 1 : 0;
                yield return m_dvrKeysCmdPassed +=
                    DVRKeysCmd(i, true, typeof(int), i) ? 1 : 0;
                yield return m_dvrKeysCmdPassed +=
                    DVRKeysCmd(i, false, typeof(int), i) ? 1 : 0;
                yield return m_dvrKeysCmdPassed +=
                    DVRKeysCmd(i, true, typeof(float), i) ? 1 : 0;
                yield return m_dvrKeysCmdPassed +=
                    DVRKeysCmd(i, false, typeof(float), i) ? 1 : 0;
            }

            m_dvrTotalTestsPassed += m_dvrKeysCmdPassed;
            if (m_dvrKeysCmdPassed == k_cmdKeysTestsExpected)
                Debug.Log("96 / 96 DVR command buffer keys tests passed.");
            else
                Debug.LogError(m_dvrKeysCmdPassed + " / 96 DVR command buffer keys tests failed.");
        }

        private IEnumerator DVRPairsStaticTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_dvrPairsStaticPassed +=
                    DVRPairsStatic(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsStaticPassed +=
                    DVRPairsStatic(i, false, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsStaticPassed +=
                    DVRPairsStatic(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsStaticPassed +=
                    DVRPairsStatic(i, false, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsStaticPassed +=
                    DVRPairsStatic(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsStaticPassed +=
                    DVRPairsStatic(i, false, typeof(int), typeof(uint), i) ? 1 : 0;
            }

            m_dvrTotalTestsPassed += m_dvrPairsStaticPassed;
            if (m_dvrPairsStaticPassed == k_staticPairsTestsExpected)
                Debug.Log("96 / 96 DVR static pairs tests passed.");
            else
                Debug.LogError(m_dvrPairsStaticPassed + " / 96 DVR static pairs tests failed.");
        }

        private IEnumerator DVRPairsCmdTest()
        {
            for (int i = 256 * k_pairsPartitionSize; i < 257 * k_pairsPartitionSize; ++i)
            {
                yield return m_dvrPairsCmdPassed +=
                    DVRPairsCmd(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsCmdPassed +=
                    DVRPairsCmd(i, false, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsCmdPassed +=
                    DVRPairsCmd(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsCmdPassed +=
                    DVRPairsCmd(i, false, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsCmdPassed +=
                    DVRPairsCmd(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return m_dvrPairsCmdPassed +=
                    DVRPairsCmd(i, false, typeof(float), typeof(uint), i) ? 1 : 0;
            }

            m_dvrTotalTestsPassed += m_dvrPairsCmdPassed;
            if (m_dvrPairsCmdPassed == k_cmdPairsTestsExpected)
                Debug.Log("46080 / 46080 DVR command buffer pairs tests passed.");
            else
                Debug.LogError(m_dvrPairsCmdPassed + " / 46080 DVR command buffer pairs tests failed.");
        }

        private IEnumerator OSResizeTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_osResizePassed +=
                    OSKeys(i, true, typeof(uint), i) ? 1 : 0;
                yield return m_osResizePassed +=
                    OSKeys(i, true, typeof(int), i) ? 1 : 0;
                yield return m_osResizePassed +=
                    OSKeys(i, true, typeof(float), i) ? 1 : 0;
            }

            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_osResizePassed +=
                    OSPairs(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_osResizePassed +=
                    OSPairs(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_osResizePassed +=
                    OSPairs(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
            }

            m_osTotalTestsPassed += m_osResizePassed;
            if (m_osResizePassed == k_resizeTestsExpected)
                Debug.Log("96 / 96 All OS resize tests passed.");
            else
                Debug.LogError(m_osResizePassed + " / 96 OS resize test failed.");
        }

        private IEnumerator OSKeysStaticTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_osKeysStaticPassed +=
                    OSKeysStatic(i, true, typeof(uint), i) ? 1 : 0;
                yield return m_osKeysStaticPassed +=
                    OSKeysStatic(i, false, typeof(uint), i) ? 1 : 0;
                yield return m_osKeysStaticPassed +=
                    OSKeysStatic(i, true, typeof(int), i) ? 1 : 0;
                yield return m_osKeysStaticPassed +=
                    OSKeysStatic(i, false, typeof(int), i) ? 1 : 0;
                yield return m_osKeysStaticPassed +=
                    OSKeysStatic(i, true, typeof(float), i) ? 1 : 0;
                yield return m_osKeysStaticPassed +=
                    OSKeysStatic(i, false, typeof(float), i) ? 1 : 0;
            }

            m_osTotalTestsPassed += m_osKeysStaticPassed;
            if (m_osKeysStaticPassed == k_staticKeysOnlyTestsExpected)
                Debug.Log("96 / 96 OS static keys tests passed.");
            else
                Debug.LogError(m_osKeysStaticPassed + " / 96 OS static keys tests failed.");
        }

        private IEnumerator OSKeysCmdTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_osKeysCmdPassed +=
                    OSKeysCmd(i, true, typeof(uint), i) ? 1 : 0;
                yield return m_osKeysCmdPassed +=
                    OSKeysCmd(i, false, typeof(uint), i) ? 1 : 0;
                yield return m_osKeysCmdPassed +=
                    OSKeysCmd(i, true, typeof(int), i) ? 1 : 0;
                yield return m_osKeysCmdPassed +=
                    OSKeysCmd(i, false, typeof(int), i) ? 1 : 0;
                yield return m_osKeysCmdPassed +=
                    OSKeysCmd(i, true, typeof(float), i) ? 1 : 0;
                yield return m_osKeysCmdPassed +=
                    OSKeysCmd(i, false, typeof(float), i) ? 1 : 0;
            }

            m_osTotalTestsPassed += m_osKeysCmdPassed;
            if (m_osKeysCmdPassed == k_cmdKeysTestsExpected)
                Debug.Log("96 / 96 OS command keys tests passed.");
            else
                Debug.LogError(m_osKeysCmdPassed + " / 96 OS command keys tests passed.");
        }

        private IEnumerator OSPairsStaticTest()
        {
            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return m_osPairsStaticPassed +=
                    OSPairsStatic(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsStaticPassed +=
                    OSPairsStatic(i, false, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsStaticPassed +=
                    OSPairsStatic(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsStaticPassed +=
                    OSPairsStatic(i, false, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsStaticPassed +=
                    OSPairsStatic(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsStaticPassed +=
                    OSPairsStatic(i, false, typeof(int), typeof(uint), i) ? 1 : 0;
            }

            m_osTotalTestsPassed += m_osPairsStaticPassed;
            if (m_osPairsStaticPassed == k_staticPairsTestsExpected)
                Debug.Log("96 / 96 OS static pairs tests passed.");
            else
                Debug.LogError(m_osPairsStaticPassed + " / 96 OS static pairs tests failed.");
        }

        private IEnumerator OSPairsCmdTest()
        {
            for (int i = 256 * k_pairsPartitionSize; i < 257 * k_pairsPartitionSize; ++i)
            {
                yield return m_osPairsCmdPassed +=
                    OSPairsCmd(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsCmdPassed +=
                    OSPairsCmd(i, false, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsCmdPassed +=
                    OSPairsCmd(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsCmdPassed +=
                    OSPairsCmd(i, false, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsCmdPassed +=
                    OSPairsCmd(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return m_osPairsCmdPassed +=
                    OSPairsCmd(i, false, typeof(float), typeof(uint), i) ? 1 : 0;
            }

            m_osTotalTestsPassed += m_osPairsCmdPassed;
            if (m_osPairsCmdPassed == k_cmdPairsTestsExpected)
                Debug.Log("46080 / 46080 OS command buffer pairs tests passed.");
            else
                Debug.LogError(m_osPairsCmdPassed + " / 46080 OS command buffer pairs tests failed.");
        }

        private IEnumerator TestAll()
        {
            m_dvr = new GPUSorting.Runtime.DeviceRadixSort(dvr);
            if (m_dvr.Valid)
                Debug.Log("DVR initialization resize success.");
            else
                Debug.LogError("DVR initialization resize failed.");
            yield return StartCoroutine(DVRResizeTest());

            m_dvr = new GPUSorting.Runtime.DeviceRadixSort(
                dvr,
                k_maxTestSize,
                ref alt,
                ref passHist,
                ref globalHist);
            if (m_dvr.Valid)
                Debug.Log("DVR initialization static keys success.");
            else
                Debug.LogError("DVR initialization static keys failed.");
            yield return StartCoroutine(DVRKeysStaticTest());
            yield return StartCoroutine(DVRKeysCmdTest());

            m_dvr = new GPUSorting.Runtime.DeviceRadixSort(
                dvr,
                k_maxTestSize,
                ref alt,
                ref altPayload,
                ref passHist,
                ref globalHist);
            if (m_dvr.Valid)
                Debug.Log("DVR initialization static pairs success.");
            else
                Debug.LogError("DVR initialization static pairs failed.");
            yield return StartCoroutine(DVRPairsStaticTest());
            yield return StartCoroutine(DVRPairsCmdTest());

            if (m_dvrTotalTestsPassed == k_totalTestsExpected)
                Debug.Log("ALL DEVICE_RADIX_SORT TESTS PASSED");
            else
                Debug.LogError("DEVICE_RADIX_SORT TESTS FAILED");

            m_oneSweep = new GPUSorting.Runtime.OneSweep(oneSweep);
            if (m_oneSweep.Valid)
                Debug.Log("OS initialization resize success.");
            else
                Debug.LogError("OS initialization resize failed.");
            yield return StartCoroutine(OSResizeTest());

            m_oneSweep = new GPUSorting.Runtime.OneSweep(
                oneSweep,
                k_maxTestSize,
                ref alt,
                ref passHist,
                ref globalHist,
                ref index);
            if (m_oneSweep.Valid)
                Debug.Log("OS initialization static keys success.");
            else
                Debug.LogError("OS initialization static keys failed.");
            yield return StartCoroutine(OSKeysStaticTest());
            yield return StartCoroutine(OSKeysCmdTest());

            m_oneSweep = new GPUSorting.Runtime.OneSweep(
                oneSweep,
                k_maxTestSize,
                ref alt,
                ref altPayload,
                ref passHist,
                ref globalHist,
                ref index);
            if (m_oneSweep.Valid)
                Debug.Log("OS initialization static pairs success.");
            else
                Debug.LogError("OS initialization static pairs failed.");
            yield return StartCoroutine(OSPairsStaticTest());
            yield return StartCoroutine(OSPairsCmdTest());

            if (m_osTotalTestsPassed == k_totalTestsExpected)
                Debug.Log("ALL ONESWEEP TESTS PASSED");
            else
                Debug.LogError("ONESWEEP TESTS FAILED");

            Debug.Log("TESTING FINISHED");
        }

        private void OnDestroy()
        {
            m_cmd?.Dispose();
            toTest?.Dispose();
            toTestPayload?.Dispose();
            alt?.Dispose();
            altPayload?.Dispose();
            globalHist?.Dispose();
            passHist?.Dispose();
            errCount?.Dispose();
            index?.Dispose();
        }
    }
}