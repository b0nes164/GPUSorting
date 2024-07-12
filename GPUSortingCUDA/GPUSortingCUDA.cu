/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include "Sort/OneSweepDispatcher.cuh"
#include "Sort/DeviceRadixSortDispatcher.cuh"
#include "Sort/CubDispatcher.cuh"
#include "Sort/EmulatedDeadlockingDispatcher.cuh"
#include "SegSort/SplitSortTests.cuh"

int main()
{
    //-----------------GLOBAL SORTS-----------------
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
        
    //-----------------SEGMETED SORT-----------------
    printf("----------------BEGINNING SEGMENTED SORT TESTS----------------\n\n");
    SplitSortTests<uint32_t>* splitSort = new SplitSortTests<uint32_t>(1 << 27, 1 << 27, 1U);
    splitSort->TestAllRandomSegmentLengths<32>(100, false);
    splitSort->~SplitSortTests();

    return 0;
}