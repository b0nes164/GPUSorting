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