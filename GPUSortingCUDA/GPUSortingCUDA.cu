#pragma once
#include "OneSweepDispatcher.cuh"
#include "DeviceRadixSortDispatcher.cuh"
#include "CubDispatcher.cuh"

int main()
{
    OneSweepDispatcher* oneSweep = new OneSweepDispatcher(1 << 28);
    oneSweep->TestAll();
    oneSweep->BatchTiming(1 << 28, 50, 10);
    oneSweep->~OneSweepDispatcher();

    DeviceRadixSortDispatcher* dvr = new DeviceRadixSortDispatcher(1 << 28);
    dvr->TestAll();
    dvr->BatchTiming(1 << 28, 50, 10);
    dvr->~DeviceRadixSortDispatcher();

    CubDispatcher* cub = new CubDispatcher(1 << 28);
    cub->BatchTimingCubDeviceRadixSort(1 << 28, 50, 10);
    cub->BatchTimingCubOneSweep(1 << 28, 50, 10);
}