/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/28/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
using System.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using GPUSorting.Runtime;

namespace GPUSorting.Tests
{
    public abstract class TestBase : MonoBehaviour
    {
        [SerializeField]
        protected ComputeShader m_util;

        protected LocalKeyword m_keyIntKeyword;
        protected LocalKeyword m_keyUintKeyword;
        protected LocalKeyword m_keyFloatKeyword;
        protected LocalKeyword m_payloadIntKeyword;
        protected LocalKeyword m_payloadUintKeyword;
        protected LocalKeyword m_payloadFloatKeyword;
        protected LocalKeyword m_ascendKeyword;
        protected LocalKeyword m_sortPairKeyword;

        protected ComputeBuffer toTest;
        protected ComputeBuffer toTestPayload;
        protected ComputeBuffer alt;
        protected ComputeBuffer altPayload;
        protected ComputeBuffer globalHist;
        protected ComputeBuffer passHist;
        protected ComputeBuffer index;
        protected ComputeBuffer errCount;

        protected CommandBuffer m_cmd;

        protected const int k_partitionSize = 3840;
        protected const int k_validatePartSize = 4096;
        protected const int k_maxTestSize = 65535 * k_partitionSize - 1;
        protected const int k_totalTestsExpected = 16 * 6 + k_partitionSize * 6 + 1;

        protected bool m_isValid;
        protected int m_kernelInitRandom = -1;
        protected int m_kernelClearErrors = -1;
        protected int m_kernelValidate = -1;

        protected string m_sortName = "";
        protected int m_totalTestPassed = 0;

        protected void Initialize()
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

            Assert.IsTrue(m_isValid);
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

        protected static int DivRoundUp(int x, int y)
        {
            return (x + y - 1) / y;
        }

        protected void InitializeTestResources()
        {
            toTest = new ComputeBuffer(k_maxTestSize, sizeof(uint));
            toTestPayload = new ComputeBuffer(k_maxTestSize, sizeof(uint));
            errCount = new ComputeBuffer(1, sizeof(uint));
        }

        protected void SetKeyTypeKeywords(System.Type _type)
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

        protected void SetPayloadTypeKeywords(System.Type _type)
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

        protected void SetAscendingKeyWords(bool _shouldAscend)
        {
            if (_shouldAscend)
                m_util.EnableKeyword(m_ascendKeyword);
            else
                m_util.DisableKeyword(m_ascendKeyword);
        }

        protected void PreSort(int _testSize, int _seed, bool keysOnly)
        {
            m_util.SetInt("e_numKeys", _testSize);
            m_util.SetInt("e_seed", _seed);

            m_util.SetBuffer(m_kernelInitRandom, "b_sort", toTest);
            if (keysOnly == false)
                m_util.SetBuffer(m_kernelInitRandom, "b_sortPayload", toTestPayload);
            m_util.Dispatch(m_kernelInitRandom, 256, 1, 1);
        }

        protected bool PostSort(int _testSize, bool keysOnly, bool shouldPrint)
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

        protected virtual bool KeysTest(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            int seed)
        {
            return false;
        }

        protected virtual bool KeysCmdTest(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            int seed)
        {
            return false;
        }

        protected virtual bool PairsTest(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
            int seed)
        {
            return false;
        }

        protected virtual bool PairsCmdTest(
            int testSize,
            bool shouldAscend,
            System.Type keyType,
            System.Type payloadType,
            int seed)
        {
            return false;
        }

        //Do ligh testing to make sure the dispatching is functioning correctly.
        protected IEnumerator KeysTestCoroutine()
        {
            int testsPassed = 0;
            const int testsExpected = 96;

            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return testsPassed +=
                    KeysTest(i, true, typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysTest(i, false, typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysTest(i, true, typeof(int), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysTest(i, false, typeof(int), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysTest(i, true, typeof(float), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysTest(i, false, typeof(float), i) ? 1 : 0;
            }

            m_totalTestPassed += testsPassed;
            if (testsPassed == testsExpected)
                Debug.Log(testsExpected + " / " + testsExpected + " " + m_sortName + " keys tests passed.");
            else
                Debug.LogError(testsPassed + " / " + testsExpected + " " + m_sortName + " keys tests failed.");
        }

        //Perform the intensive testing
        protected IEnumerator KeysCmdTestCoroutine()
        {
            Debug.Log("Beggining Cmd Tests, this may take a while.");

            int testsPassed = 0;
            const int testsExpected = k_partitionSize * 6 + 1;

            for (int i = 512 * k_partitionSize; i < 513 * k_partitionSize; ++i)
            {
                yield return testsPassed +=
                    KeysCmdTest(i, true, typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysCmdTest(i, false, typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysCmdTest(i, true, typeof(int), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysCmdTest(i, false, typeof(int), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysCmdTest(i, true, typeof(float), i) ? 1 : 0;
                yield return testsPassed +=
                    KeysCmdTest(i, false, typeof(float), i) ? 1 : 0;

                Debug.Log("Running.");
            }

            //One max size test
            yield return testsPassed +=
                    KeysCmdTest(k_maxTestSize, true, typeof(uint), 10) ? 1 : 0;

            m_totalTestPassed += testsPassed;
            if (testsPassed == testsExpected)
                Debug.Log(testsExpected + " / " + testsExpected + " " + m_sortName + " keys cmd tests passed.");
            else
                Debug.LogError(testsPassed + " / " + testsExpected + " " + m_sortName + " keys cmd tests failed.");
        }

        protected IEnumerator PairsTestCoroutine()
        {
            int testsPassed = 0;
            const int testsExpected = 96;

            for (int i = 2; i <= 65536; i <<= 1)
            {
                yield return testsPassed +=
                    PairsTest(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsTest(i, false, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsTest(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsTest(i, false, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsTest(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsTest(i, false, typeof(int), typeof(uint), i) ? 1 : 0;
            }

            m_totalTestPassed += testsPassed;
            if (testsPassed == testsExpected)
                Debug.Log(testsExpected + " / " + testsExpected + " " + m_sortName + " pairs tests passed.");
            else
                Debug.LogError(testsPassed + " / " + testsExpected + " " + m_sortName + " pairs tests failed.");
        }

        protected IEnumerator PairsCmdTestCoroutine()
        {
            Debug.Log("Beggining Cmd Tests, this may take a while.");

            int testsPassed = 0;
            const int testsExpected = k_partitionSize * 6 + 1;

            for (int i = 512 * k_partitionSize; i < 513 * k_partitionSize; ++i)
            {
                yield return testsPassed +=
                    PairsCmdTest(i, true, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsCmdTest(i, false, typeof(uint), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsCmdTest(i, true, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsCmdTest(i, false, typeof(int), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsCmdTest(i, true, typeof(float), typeof(uint), i) ? 1 : 0;
                yield return testsPassed +=
                    PairsCmdTest(i, false, typeof(float), typeof(uint), i) ? 1 : 0;

                Debug.Log("Running.");
            }

            //One max size test
            yield return testsPassed +=
                PairsCmdTest(k_maxTestSize, true, typeof(uint), typeof(uint), 10) ? 1 : 0;

            m_totalTestPassed += testsPassed;
            if (testsPassed == testsExpected)
                Debug.Log(testsExpected + " / " + testsExpected + " " + m_sortName + " pairs cmd tests passed.");
            else
                Debug.LogError(testsPassed + " / " + testsExpected + " " + m_sortName + " pairs cmd tests failed.");
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
