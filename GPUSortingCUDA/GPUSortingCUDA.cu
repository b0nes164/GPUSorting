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
#include "SegSort/SplitSortDispatcher.cuh"
#include "SegSort/SplitSortVariantTester.cuh"

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

    EmulatedDeadlockingDispatcher* d = new EmulatedDeadlockingDispatcher(1 << 28);
    d->BatchTimingKeysOnly(1 << 28, 100, 10, ENTROPY_PRESET_1);
    d->~EmulatedDeadlockingDispatcher();
        
    //-----------------SEGMETED SORT-----------------
    SplitSortDispatcher<uint32_t>* sp = new SplitSortDispatcher<uint32_t>(1 << 27, 1 << 27, 1U);
    sp->TestAllRandomSegmentLengths<32>(50, false);
    sp->~SplitSortDispatcher();

    //-----------------SEGMENT VARIANTS-----------------
    SplitSortVariantTester* spv = new SplitSortVariantTester(1 << 27, 1 << 22);
    spv->Dispatch();

    /*
    //32
    spv->BatchTime_w1_t32_kv32_cute32_bin(100, 1 << 22, 32);
    spv->BatchTime_w2_t32_kv32_cute32_bin(100, 1 << 22, 32);              
    spv->BatchTime_w4_t32_kv32_cute32_bin(100, 1 << 22, 32);            //Best 32
   
    //64
    spv->BatchTime_w1_t32_kv64_cute32_wMerge(100, 1 << 21, 64);
    spv->BatchTime_w2_t32_kv64_cute32_wMerge(100, 1 << 21, 64);
    spv->BatchTime_w4_t32_kv64_cute32_wMerge(100, 1 << 21, 64);         //Best 64

    spv->BatchTime_w1_t32_kv64_cute64_wMerge(100, 1 << 21, 64);
    spv->BatchTime_w2_t32_kv64_cute64_wMerge(100, 1 << 21, 64);
    spv->BatchTime_w4_t32_kv64_cute64_wMerge(100, 1 << 21, 64);
    
    //128
    spv->BatchTime_w1_t32_kv128_cute32_wMerge(100, 1 << 20, 128);
    spv->BatchTime_w2_t32_kv128_cute32_wMerge(100, 1 << 20, 128);
    spv->BatchTime_w4_t32_kv128_cute32_wMerge(100, 1 << 20, 128);
    
    spv->BatchTime_w1_t32_kv128_cute64_wMerge(100, 1 << 20, 128);
    spv->BatchTime_w2_t32_kv128_cute64_wMerge(100, 1 << 20, 128);       //Best 128
    spv->BatchTime_w4_t32_kv128_cute64_wMerge(100, 1 << 20, 128);        

    spv->BatchTime_w2_t32_kv128_cute128_wMerge(100, 1 << 20, 128);

    spv->BatchTime_w1_t64_kv128_cute32_bMerge(100, 1 << 20, 128);
    spv->BatchTime_w1_t64_kv128_cute64_bMerge(100, 1 << 20, 128);

    //256
    spv->BatchTime_w2_t32_kv256_cute32_wMerge(100, 1 << 19, 256);
    spv->BatchTime_w2_t32_kv256_cute64_wMerge(100, 1 << 19, 256);
    spv->BatchTime_w2_t32_kv256_cute128_wMerge(100, 1 << 19, 256);
    spv->BatchTime_w1_t64_kv256_cute32_bMerge(100, 1 << 19, 256);
    spv->BatchTime_w1_t64_kv256_cute64_bMerge(100, 1 << 19, 256);       //Best 256
    spv->BatchTime_w1_t64_kv256_cute128_bMerge(100, 1 << 19, 256);
    spv->BatchTime_w1_t128_kv256_cute32_bMerge(100, 1 << 19, 256);      
    spv->BatchTime_w1_t128_kv256_cute64_bMerge(100, 1 << 19, 256);
    spv->BatchTime_w1_t256_kv256_cute32_bMerge(100, 1 << 19, 256);
    
    //512
    spv->BatchTime_w1_t64_kv512_cute32_bMerge(100, 1 << 18, 512);
    spv->BatchTime_w1_t128_kv512_cute32_bMerge(100, 1 << 18, 512);
    spv->BatchTime_w1_t256_kv512_cute32_bMerge(100, 1 << 18, 512);

    spv->BatchTime_w1_t64_kv512_cute64_bMerge(100, 1 << 18, 512);
    spv->BatchTime_w1_t128_kv512_cute64_bMerge(100, 1 << 18, 512);
    spv->BatchTime_w1_t256_kv512_cute64_bMerge(100, 1 << 18, 512);

    spv->BatchTime_w1_t64_kv512_cute128_bMerge(100, 1 << 18, 512);
    spv->BatchTime_w1_t128_kv512_cute128_bMerge(100, 1 << 18, 512);     //Best 512
    
    //1024
    spv->BatchTime_w1_t128_kv1024_cute64_bMerge(100, 1 << 17, 1024);
    spv->BatchTime_w1_t256_kv1024_cute64_bMerge(100, 1 << 17, 1024);
    spv->BatchTime_w1_t128_kv1024_cute128_bMerge(100, 1 << 17, 1024);
    spv->BatchTime_w1_t256_kv1024_cute128_bMerge(100, 1 << 17, 1024);   //Best 1024
    
    //2048
    spv->BatchTime_w1_t512_kv2048_cute64_bMerge(100, 1 << 16, 2048);
    spv->BatchTime_w1_t512_kv2048_cute128_bMerge(100, 1 << 16, 2048);
    spv->BatchTime_w1_t1024_kv2048_cute64_bMerge(100, 1 << 16, 2048);

    //RADIX, 128
    spv->BatchTime_w1_t64_kv128_radix(100, 1 << 20, 128);

    //256
    spv->BatchTime_w1_t64_kv256_radix(100, 1 << 19, 256);
    
    //512
    spv->BatchTime_w1_t64_kv512_radix(100, 1 << 18, 512);
    spv->BatchTime_w1_t128_kv512_radix(100, 1 << 18, 512);
    spv->BatchTime_w1_t256_kv512_radix(100, 1 << 18, 512);
    
    //1024
    spv->BatchTime_w1_t128_kv1024_radix(100, 1 << 17, 1024);            
    spv->BatchTime_w1_t256_kv1024_radix(100, 1 << 17, 1024);

    //2048
    spv->BatchTime_w1_t256_kv2048_radix(100, 1 << 16, 2048);            //Best 2048
    spv->BatchTime_w1_t512_kv2048_radix(100, 1 << 16, 2048);
    
    //4096
    spv->BatchTime_w1_t512_kv4096_radix(100, 1 << 15, 4096);            //Only available 4096
    */
    spv->~SplitSortVariantTester();
    
    return 0;
}