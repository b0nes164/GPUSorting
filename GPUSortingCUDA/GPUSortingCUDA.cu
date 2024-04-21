/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include "OneSweepDispatcher.cuh"
#include "DeviceRadixSortDispatcher.cuh"
#include "CubDispatcher.cuh"
#include "EmulatedDeadlockingDispatcher.cuh"

int main()
{
    printf("-----------------BEGINNING KEYS TESTS-----------------\n\n");
    OneSweepDispatcher* oneSweep = new OneSweepDispatcher(true, 1 << 28);
    oneSweep->TestAllKeysOnly();
    oneSweep->BatchTimingKeysOnly(1 << 28, 100, 10, ENTROPY_PRESET_1);
    oneSweep->~OneSweepDispatcher();

    DeviceRadixSortDispatcher* dvr = new DeviceRadixSortDispatcher(true, 1 << 28);
    dvr->TestAllKeysOnly();
    dvr->BatchTimingKeysOnly(1 << 28, 100, 10, ENTROPY_PRESET_1);
    dvr->~DeviceRadixSortDispatcher();

    CubDispatcher* cub = new CubDispatcher(true, 1 << 28);
    cub->BatchTimingCubDeviceRadixSortKeys(1 << 28, 100, 10, ENTROPY_PRESET_1);
    cub->BatchTimingCubOneSweepKeys(1 << 28, 100, 10, ENTROPY_PRESET_1);
    cub->~CubDispatcher();

    printf("----------------BEGINNING PAIRS TESTS----------------\n\n");
    oneSweep = new OneSweepDispatcher(false, 1 << 28);
    oneSweep->TestAllPairs();
    oneSweep->BatchTimingPairs(1 << 28, 100, 10, ENTROPY_PRESET_1);
    oneSweep->~OneSweepDispatcher();

    dvr = new DeviceRadixSortDispatcher(false, 1 << 28);
    dvr->TestAllPairs();
    dvr->BatchTimingPairs(1 << 28, 100, 10, ENTROPY_PRESET_1);
    dvr->~DeviceRadixSortDispatcher();

    cub = new CubDispatcher(false, 1 << 28);
    cub->BatchTimingCubDeviceRadixSortPairs(1 << 28, 100, 10, ENTROPY_PRESET_1);
    cub->BatchTimingCubOneSweepPairs(1 << 28, 100, 10, ENTROPY_PRESET_1);
    cub->~CubDispatcher();

    return 0;
}