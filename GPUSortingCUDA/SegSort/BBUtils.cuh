/******************************************************************************
* BBUtils
* Utilities and functions used from bb_segsort
* 
* bb_segsort:
*       Kaixi Hou
*       Weifeng Lie
*       Hao Wang
*       Wu-chun Feng
*       https://github.com/vtsynergy/bb_segsort
* 
* (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
 ******************************************************************************/
#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CMP_SWP(t1,_a,_b,t2,_c,_d) if(_a>_b)  {t1 _t=_a;_a=_b;_b=_t; t2 _s=_c;_c=_d;_d=_s;}
#define EQL_SWP(t1,_a,_b,t2,_c,_d) if(_a!=_b) {t1 _t=_a;_a=_b;_b=_t; t2 _s=_c;_c=_d;_d=_s;}
#define     SWP(t1,_a,_b,t2,_c,_d)            {t1 _t=_a;_a=_b;_b=_t; t2 _s=_c;_c=_d;_d=_s;}

namespace BBUtils
{
    template<class K>
    __device__ __forceinline__ void exch_intxn(K& k0, uint32_t& v0, int mask, const int bit) {
        K ex_k0, ex_k1;
        uint32_t ex_v0, ex_v1;
        ex_k0 = k0;
        ex_k1 = __shfl_xor_sync(0xffffffff, k0, mask);
        ex_v0 = v0;
        ex_v1 = __shfl_xor_sync(0xffffffff, v0, mask);
        CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
        if (bit) EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
        k0 = ex_k0;
        v0 = ex_v0;
    }

    template<class K>
    __device__ inline void exch_paral(K& k0, uint32_t& v0, int mask, const int bit) {
        K ex_k0, ex_k1;
        uint32_t ex_v0, ex_v1;
        ex_k0 = k0;
        ex_k1 = __shfl_xor_sync(0xffffffff, k0, mask);
        ex_v0 = v0;
        ex_v1 = __shfl_xor_sync(0xffffffff, v0, mask);
        CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
        if (bit) EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
        k0 = ex_k0;
        v0 = ex_v0;
    }

    //2 keys
    template<class K>
    __device__ __forceinline__ void t2_kv2(
        K& key,
        uint32_t& index)
    {
        exch_intxn(key, index, 0x1, getLaneId() & 1);
    }

    //4 keys
    template<class K>
    __device__ __forceinline__ void t4_kv4(
        K& key,
        uint32_t& index)
    {
        const uint32_t tid = getLaneId() & 3;
        const uint32_t bit1 = tid & 1;
        exch_intxn(key, index, 0x1, bit1);
        exch_intxn(key, index, 0x3, tid >> 1 & 1);
        exch_paral(key, index, 0x1, bit1);
    }

    //8 keys
    template<class K>
    __device__ __forceinline__ void t8_kv8(
        K& key,
        uint32_t& index)
    {
        const uint32_t tid = getLaneId() & 7;
        const uint32_t bit1 = tid & 1;
        const uint32_t bit2 = tid >> 1 & 1;
        exch_intxn(key, index, 0x1, bit1);
        exch_intxn(key, index, 0x3, bit2);
        exch_paral(key, index, 0x1, bit1);
        exch_intxn(key, index, 0x7, tid >> 2 & 1);
        exch_paral(key, index, 0x2, bit2);
        exch_paral(key, index, 0x1, bit1);
    }

    //16 keys
    template<class K>
    __device__ __forceinline__ void t16_kv16(
        K& key,
        uint32_t& index)
    {
        const uint32_t tid = getLaneId() & 15;
        const uint32_t bit1 = tid & 1;
        const uint32_t bit2 = tid >> 1 & 1;
        const uint32_t bit3 = tid >> 2 & 1;

        exch_intxn(key, index, 0x1, bit1);
        exch_intxn(key, index, 0x3, bit2);
        exch_paral(key, index, 0x1, bit1);
        exch_intxn(key, index, 0x7, bit3);
        exch_paral(key, index, 0x2, bit2);
        exch_paral(key, index, 0x1, bit1);
        exch_intxn(key, index, 0xf, tid >> 3 & 1);
        exch_paral(key, index, 0x4, bit3);
        exch_paral(key, index, 0x2, bit2);
        exch_paral(key, index, 0x1, bit1);
    }

    template<class K>
    __device__ __forceinline__ void RegSortFallback(
        K& key,
        uint32_t& index,
        const uint32_t sortLength)
    {
        if (sortLength < 5)
        {
            if (sortLength < 3)
            {
                if (sortLength == 2)
                    t2_kv2(key, index);
            }
            else
            {
                t4_kv4(key, index);
            }
        }
        else
        {
            if (sortLength < 9)
                t8_kv8(key, index);
            else
                t16_kv16(key, index);
        }
    }

    __device__ __forceinline__ uint2 BinarySearch(
        const uint32_t* s_warpBins,
        const int32_t binCount,
        const uint32_t targetIndex)
    {
        const uint32_t start = s_warpBins[0];
        if (binCount > 1)
        {
            const uint32_t t = start + targetIndex;
            int32_t l = 0;
            int32_t h = binCount;

            while (l < h)
            {
                const int32_t m = l + (h - l) / 2;
                if (m >= binCount)  //Unnecessary?
                    break;

                const uint32_t lr = s_warpBins[m];
                const uint32_t rr = s_warpBins[m + 1];
                if (lr <= t && t < rr)
                    return { lr - start, rr - start };
                else if (t < lr)
                    h = m;
                else
                    l = m + 1;
            }

            return { 0, 0 };
        }
        else
        {
            return { 0, s_warpBins[1] - start };
        }
    }

    __device__ __forceinline__ uint32_t find_kth3(
        const uint2* a,
        const uint2* b,
        const int length,
        const int diag)
    {
        int begin = max(0, diag - length);
        int end = min(diag, length);

        while (begin < end) {
            int mid = (begin + end) >> 1;
            uint32_t aKey = a[mid].x;
            uint32_t bKey = b[diag - 1 - mid].x;
            bool pred = aKey <= bKey;
            if (pred) begin = mid + 1;
            else end = mid;
        }
        return begin;
    }
}