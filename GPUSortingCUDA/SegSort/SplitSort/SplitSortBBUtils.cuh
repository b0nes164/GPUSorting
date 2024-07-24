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
#include "SplitSortUtils.cuh"

#define CMP_SWPS(_a,_b,_c,_d) if(_a>_b) {K _t=_a;_a=_b;_b=_t; V _s=_c;_c=_d;_d=_s;}
#define EQL_SWPS(_a,_b,_c,_d) if(_a!=_b) {K _t=_a;_a=_b;_b=_t; V _s=_c;_c=_d;_d=_s;}

namespace SplitSortInternal
{
    template<class K, class V>
    __device__ __forceinline__ void exch_intxn(K& k0, V& v0, int mask, const int bit)
    {
        K ex_k0, ex_k1;
        V ex_v0, ex_v1;
        ex_k0 = k0;
        ex_k1 = __shfl_xor_sync(0xffffffff, k0, mask);
        ex_v0 = v0;
        ex_v1 = __shfl_xor_sync(0xffffffff, v0, mask);
        CMP_SWPS(ex_k0, ex_k1, ex_v0, ex_v1);
        if (bit)
            EQL_SWPS(ex_k0, ex_k1, ex_v0, ex_v1);
        k0 = ex_k0;
        v0 = ex_v0;
    }

    //Exactly the same as exch_paral?
    template<class K, class V>
    __device__ inline void exch_paral(K& k0, V& v0, int mask, const int bit) 
    {
        K ex_k0, ex_k1;
        V ex_v0, ex_v1;
        ex_k0 = k0;
        ex_k1 = __shfl_xor_sync(0xffffffff, k0, mask);
        ex_v0 = v0;
        ex_v1 = __shfl_xor_sync(0xffffffff, v0, mask);
        CMP_SWPS(ex_k0, ex_k1, ex_v0, ex_v1);
        if (bit)
            EQL_SWPS(ex_k0, ex_k1, ex_v0, ex_v1);
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

    __device__ __forceinline__ uint32_t find_kth3_device(
        const uint32_t* a,
        const uint32_t* b,
        const int length,
        const int diag)
    {
        int begin = max(0, diag - length);
        int end = min(diag, length);

        while (begin < end) {
            int mid = (begin + end) >> 1;
            uint32_t aKey = a[mid];
            uint32_t bKey = b[diag - 1 - mid];
            bool pred = aKey <= bKey;
            if (pred) begin = mid + 1;
            else end = mid;
        }
        return begin;
    }

    __device__ __forceinline__ uint32_t find_kth3_device_partial(
        const uint32_t* a,
        const uint32_t* b,
        const int length,
        const int diag,
        const int topLength)
    {
        int begin = max(0, diag - length);
        int end = min(diag, length);

        while (begin < end) {
            int mid = (begin + end) >> 1;
            uint32_t aKey = a[mid];
            int top = diag - 1 - mid;
            uint32_t bKey = top < topLength ? b[top] : 0xffffffff;
            bool pred = aKey <= bKey;
            if (pred) begin = mid + 1;
            else end = mid;
        }
        return begin;
    }

    template<class T>
    __device__ __forceinline__ void TransposeAndWrite8(
        T* t,
        T* dest,
        const uint32_t mergeId,
        const uint32_t remainingLength)
    {
        const uint32_t lane_id = getLaneId();
        t[1] = __shfl_xor_sync(0xffffffff, t[1], 0x1);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x1);
        t[5] = __shfl_xor_sync(0xffffffff, t[5], 0x1);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x1);
        if (lane_id & 0x1)
        {
            T temp = t[0];
            t[0] = t[1];
            t[1] = temp;
            temp = t[2];
            t[2] = t[3];
            t[3] = temp;
            temp = t[4];
            t[4] = t[5];
            t[5] = temp;
            temp = t[6];
            t[6] = t[7];
            t[7] = temp;
        }
        t[1] = __shfl_xor_sync(0xffffffff, t[1], 0x1);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x1);
        t[5] = __shfl_xor_sync(0xffffffff, t[5], 0x1);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x1);
        t[2] = __shfl_xor_sync(0xffffffff, t[2], 0x2);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x2);
        t[6] = __shfl_xor_sync(0xffffffff, t[6], 0x2);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x2);
        if (lane_id & 0x2)
        {
            T temp = t[0];
            t[0] = t[2];
            t[2] = temp;
            temp = t[1];
            t[1] = t[3];
            t[3] = temp;
            temp = t[4];
            t[4] = t[6];
            t[6] = temp;
            temp = t[5];
            t[5] = t[7];
            t[7] = temp;
        }
        t[2] = __shfl_xor_sync(0xffffffff, t[2], 0x2);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x2);
        t[6] = __shfl_xor_sync(0xffffffff, t[6], 0x2);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x2);
        t[4] = __shfl_xor_sync(0xffffffff, t[4], 0x4);
        t[5] = __shfl_xor_sync(0xffffffff, t[5], 0x4);
        t[6] = __shfl_xor_sync(0xffffffff, t[6], 0x4);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x4);
        if (lane_id & 0x4)
        {
            T temp = t[0];
            t[0] = t[4];
            t[4] = temp;
            temp = t[1];
            t[1] = t[5];
            t[5] = temp;
            temp = t[2];
            t[2] = t[6];
            t[6] = temp;
            temp = t[3];
            t[3] = t[7];
            t[7] = temp;
        }
        t[4] = __shfl_xor_sync(0xffffffff, t[4], 0x4);
        t[5] = __shfl_xor_sync(0xffffffff, t[5], 0x4);
        t[6] = __shfl_xor_sync(0xffffffff, t[6], 0x4);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x4);
        t[1] = __shfl_xor_sync(0xffffffff, t[1], 0x8);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x8);
        t[5] = __shfl_xor_sync(0xffffffff, t[5], 0x8);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x8);
        if (lane_id & 0x8)
        {
            T temp = t[0];
            t[0] = t[1];
            t[1] = temp;
            temp = t[2];
            t[2] = t[3];
            t[3] = temp;
            temp = t[4];
            t[4] = t[5];
            t[5] = temp;
            temp = t[6];
            t[6] = t[7];
            t[7] = temp;
        }
        t[1] = __shfl_xor_sync(0xffffffff, t[1], 0x8);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x8);
        t[5] = __shfl_xor_sync(0xffffffff, t[5], 0x8);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x8);
        t[2] = __shfl_xor_sync(0xffffffff, t[2], 0x10);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x10);
        t[6] = __shfl_xor_sync(0xffffffff, t[6], 0x10);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x10);
        if (lane_id & 0x10)
        {
            T temp = t[0];
            t[0] = t[2];
            t[2] = temp;
            temp = t[1];
            t[1] = t[3];
            t[3] = temp;
            temp = t[4];
            t[4] = t[6];
            t[6] = temp;
            temp = t[5];
            t[5] = t[7];
            t[7] = temp;
        }
        t[2] = __shfl_xor_sync(0xffffffff, t[2], 0x10);
        t[3] = __shfl_xor_sync(0xffffffff, t[3], 0x10);
        t[6] = __shfl_xor_sync(0xffffffff, t[6], 0x10);
        t[7] = __shfl_xor_sync(0xffffffff, t[7], 0x10);

        uint32_t offset = lane_id + (mergeId >> LANE_LOG << 8);
        if(offset < remainingLength)
            dest[offset] = t[0];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[4];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[1];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[5];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[2];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[6];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[3];
        offset += LANE_COUNT;
        if (offset < remainingLength)
            dest[offset] = t[7];
    }
}; //Semi colon stop intellisense breaking?

#undef EQL_SWPS
#undef CMP_SWPS