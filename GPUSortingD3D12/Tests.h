#pragma once
#include "pch.h"
#include "DeviceRadixSort.h"
#include "OneSweep.h"

static void SuperTestOneSweep(
    winrt::com_ptr<ID3D12Device> device,
    DeviceInfo const& deviceInfo)
{
    //KEY INT
    OneSweep* oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    //KEY FLOAT
    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    //KEY INT
    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();
}

static void SuperTestDeviceRadixSort(
    winrt::com_ptr<ID3D12Device> device,
    DeviceInfo const& deviceInfo)
{
    //KEY UINT
    DeviceRadixSort* dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_UINT32,
        GPU_SORTING_PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    //KEY FLOAT
    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_FLOAT32,
        GPU_SORTING_PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    //KEY INT
    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_ASCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPU_SORTING_DESCENDING,
        GPU_SORTING_KEY_INT32,
        GPU_SORTING_PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();
}