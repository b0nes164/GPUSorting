#pragma once
#include "pch.h"
#include "DeviceRadixSort.h"
#include "OneSweep.h"

static void SuperTestOneSweep(
    winrt::com_ptr<ID3D12Device> device,
    const GPUSorting::DeviceInfo& deviceInfo)
{
    //KEY INT
    OneSweep* oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    //KEY FLOAT
    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    //KEY INT
    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();

    oneSweep = new OneSweep(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    oneSweep->TestAll();
    oneSweep->~OneSweep();
}

static void SuperTestDeviceRadixSort(
    winrt::com_ptr<ID3D12Device> device,
    const GPUSorting::DeviceInfo& deviceInfo)
{
    //KEY UINT
    DeviceRadixSort* dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_UINT32,
        GPUSorting::PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    //KEY FLOAT
    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_FLOAT32,
        GPUSorting::PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    //KEY INT
    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_UINT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_FLOAT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_ASCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();

    dvr = new DeviceRadixSort(
        device,
        deviceInfo,
        GPUSorting::ORDER_DESCENDING,
        GPUSorting::KEY_INT32,
        GPUSorting::PAYLOAD_INT32);
    dvr->TestAll();
    dvr->~DeviceRadixSort();
}