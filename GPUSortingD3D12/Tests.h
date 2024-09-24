#pragma once
#include "pch.h"
#include "DeviceRadixSort.h"
#include "OneSweep.h"

static void SuperTestOneSweep(
    winrt::com_ptr<ID3D12Device> device,
    const GPUSorting::DeviceInfo& deviceInfo)
{
    const uint32_t testsExpected = 18;
    uint32_t testsPassed = 0;

    //KEY INT
    OneSweep* oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    //KEY FLOAT
    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    //KEY INT
    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += oneSweep->TestAll();
    oneSweep->~OneSweep();

    printf("\n");
    printf("\n---------------------------------------------------------");
    printf("\n-------------------ONESWEEP SUPER TEST-------------------");
    printf("\n---------------------------------------------------------\n");
    if (testsPassed == testsExpected)
        printf("%u / %u ONESWEEP SUPER TEST PASSED!\n", testsPassed, testsExpected);
    else
        printf("%u / %u ONESWEEP SUPER TEST FAILED!\n", testsPassed, testsExpected);
}

static void SuperTestDeviceRadixSort(
    winrt::com_ptr<ID3D12Device> device,
    const GPUSorting::DeviceInfo& deviceInfo)
{
    const uint32_t testsExpected = 18;
    uint32_t testsPassed = 0;

    //KEY UINT
    DeviceRadixSort* dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    //KEY FLOAT
    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    //KEY INT
    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    testsPassed += dvr->TestAll();
    dvr->~DeviceRadixSort();

    printf("\n");
    printf("\n----------------------------------------------------------");
    printf("\n---------------DEVICE RADIX SORT SUPER TEST---------------");
    printf("\n----------------------------------------------------------\n");
    if (testsPassed == testsExpected)
        printf("%u / %u DEVICE RADIX SORT SUPER TEST PASSED!\n", testsPassed, testsExpected);
    else
        printf("%u / %u DEVICE RADIX SORT SUPER TEST FAILED!\n", testsPassed, testsExpected);
}

static void BenchmarkOneSweep(
    winrt::com_ptr<ID3D12Device> device,
    const GPUSorting::DeviceInfo& deviceInfo)
{
    printf("---------------------------------------------------------");
    printf("\n---------------ONESWEEP KEYS ENTROPY SWEEP---------------");
    printf("\n---------------------------------------------------------\n");
    OneSweep* oneSweepKeys = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32);
    oneSweepKeys->TestAll();
    oneSweepKeys->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_1);
    oneSweepKeys->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_2);
    oneSweepKeys->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_3);
    oneSweepKeys->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_4);
    oneSweepKeys->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_5);

    printf("\n---------------------------------------------------------");
    printf("\n-----------------ONESWEEP KEYS SIZE SWEEP----------------");
    printf("\n---------------------------------------------------------\n");
    for (uint32_t i = 10; i < 28; ++i)
        oneSweepKeys->BatchTiming(1 << i, 500, 10, GPUSorting::ENTROPY_PRESET_1);
    oneSweepKeys->~OneSweep();

    printf("\n---------------------------------------------------------");
    printf("\n--------------ONESWEEP PAIRS ENTROPY SWEEP---------------");
    printf("\n---------------------------------------------------------\n");
    OneSweep* oneSweepPairs = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweepPairs->TestAll();
    oneSweepPairs->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_1);
    oneSweepPairs->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_2);
    oneSweepPairs->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_3);
    oneSweepPairs->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_4);
    oneSweepPairs->BatchTiming(1 << 28, 500, 10, GPUSorting::ENTROPY_PRESET_5);

    printf("\n---------------------------------------------------------");
    printf("\n----------------ONESWEEP PAIRS SIZE SWEEP----------------");
    printf("\n---------------------------------------------------------\n");
    for (uint32_t i = 10; i < 28; ++i)
        oneSweepPairs->BatchTiming(1 << i, 500, 10, GPUSorting::ENTROPY_PRESET_1);
    oneSweepPairs->~OneSweep();

}