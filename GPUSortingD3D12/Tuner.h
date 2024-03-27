/******************************************************************************
 * GPUSorting
 * Warning: these tables may contain inaccuracies
 * 
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 3/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#pragma once
#include "pch.h"
#include "GPUSorting.h"

namespace Tuner::TunerHelper
{
    //Nvidia, at the moment we only really need the shared memory size,
    //but in the future more of this information could be useful
    struct adapterInfoNvidia
    {
        const char* deviceName;
        uint32_t smCount;
        uint32_t registerFileSize;		//32-bit registers, not bytes
        uint32_t sharedMemoryPerSM;
        uint32_t maxWavesPerSM;
    };

    //At this point tuning is supported for RDNA1+ only
    //So we just check for RDNA1+
    struct adapterInfoAmd
    {
        const char* deviceName;
    };

    //In the future these could perfect hash tables, but unnecessary at the moment.
    static std::unordered_map<uint32_t, adapterInfoNvidia> InitializeNvidiaTuningTable()
    {
        std::unordered_map<uint32_t, adapterInfoNvidia> adapterTable
        {
            //NVIDIA
            //1000 SERIES NON_MOBILE
            { 0x1d02, {"NVIDIA GeForce GT 1010", 2, 65536, 16384, 64}},

            { 0x1d01, {"NVIDIA GeForce GT 1030", 3, 65536, 49152, 64}},

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GT 1010 DDR4", 2, 65536, 16384, 64}},

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GT 1030 DDR4", 3, 65536, 49152, 64}},

            { 0x1c81, {"NVIDIA GeForce GTX 1050", 5, 65536, 49152, 64}},

            { 0x1c82, {"NVIDIA GeForce GTX 1050 Ti", 6, 65536, 49152, 64}},

            { 0x1c83, {"NVIDIA GeForce GTX 1050 3 GB", 6, 65536, 49152, 64}},

            { 0x1c02, {"NVIDIA GeForce GTX 1060 3 GB", 9, 65536, 49152, 64}},

            { 0x1c03, {"NVIDIA GeForce GTX 1060 6 GB", 10, 65536, 49152, 64}},

            { 0x1c04, {"NVIDIA GeForce GTX 1060 5 GB", 10, 65536, 49152, 64}},

            { 0x1c06, {"NVIDIA GeForce GTX 1060 6 GB Rev. 2", 10, 65536, 49152, 64}},

            { 0x1b84, {"NVIDIA GeForce GTX 1060 3 GB GP104", 9, 65536, 49152, 64}},

            { 0x1b83, {"NVIDIA GeForce GTX 1060 6 GB GP104", 10, 65536, 49152, 64}},

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1060 6 GB 9Gbps", 10, 65536, 49152, 64}},

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1060 6 GB GDDR5X", 10, 65536, 49152, 64}},

            { 0x1b81, {"NVIDIA GeForce GTX 1070", 15, 65536, 49152, 64}},

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1070 GDDR5X", 15, 65536, 49152, 64}},

            { 0x1b82, {"NVIDIA GeForce GTX 1070 Ti", 19, 65536, 49152, 64}},

            { 0x1b80, {"NVIDIA GeForce GTX 1080", 20, 65536, 49152, 64}},

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1080 11Gbps", 20, 65536, 49152, 64}},

            { 0x1b06, {"NVIDIA GeForce GTX 1080 Ti", 28, 65536, 49152, 64}},

            { 0x1b00, {"NVIDIA TITAN X Pascal", 28, 65536, 49152, 64}},

            { 0x1b02, {"NVIDIA TITAN Xp", 30, 65536, 49152, 64}},

            //*****************************************************************************
            //*****************************************************************************
            //1000 SERIES MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x1c22, {"NVIDIA GeForce GTX 1050 Mobile GP106M", 5, 65536, 49152, 64}},

            { 0x1c21, {"NVIDIA GeForce GTX 1050 Ti Mobile GP106M", 6, 65536, 49152, 64}},

            { 0x1c62, {"NVIDIA GeForce GTX 1050 Mobile GP106BM", 5, 65536, 49152, 64}},

            { 0x1c61, {"NVIDIA GeForce GTX 1050 Ti Mobile GP106BM", 6, 65536, 49152, 64}},

            { 0x1c8d, {"NVIDIA GeForce GTX 1050 Mobile GP107M", 5, 65536, 49152, 64}},

            { 0x1c8c, {"NVIDIA GeForce GTX 1050 Ti Mobile GP107M", 6, 65536, 49152, 64}},

            { 0x1c91, {"NVIDIA GeForce GTX 1050 Max-Q", 5, 65536, 49152, 64}},

            { 0x1c8f, {"NVIDIA GeForce GTX 1050 Ti Max-Q", 6, 65536, 49152, 64}},

            { 0x1c92, {"NVIDIA GeForce GTX 1050 Mobile GP107M", 5, 65536, 49152, 64}},

            { 0x1ccd, {"NVIDIA GeForce GTX 1050 Mobile GP107BM", 5, 65536, 49152, 64}},

            { 0x1ccc, {"NVIDIA GeForce GTX 1050 Ti Mobile GP107BM", 6, 65536, 49152, 64}},

            { 0x1c20, {"NVIDIA GeForce GTX 1060 Mobile GP106M", 10, 65536, 49152, 64}},

            { 0x1c23, {"NVIDIA GeForce GTX 1060 Mobile Rev. 2 GP106M", 10, 65536, 49152, 64}},

            { 0x1c60, {"NVIDIA GeForce GTX 1060 Mobile 6 GB GP106BM", 10, 65536, 49152, 64}},

            { 0x1ba1, {"NVIDIA GeForce GTX 1070 Mobile GP104M", 16, 65536, 49152, 64}},

            { 0x1ba2, {"NVIDIA GeForce GTX 1070 Mobile GP104M", 16, 65536, 49152, 64}},

            { 0x1be1, {"NVIDIA GeForce GTX 1070 Mobile GP104BM", 16, 65536, 49152, 64}},

            { 0x1ba0, {"NVIDIA GeForce GTX 1080 Mobile GP104M", 20, 65536, 49152, 64}},

            { 0x1be0, {"NVIDIA GeForce GTX 1080 Mobile GP104BM", 20, 65536, 49152, 64}},

            //*****************************************************************************
            //*****************************************************************************
            //1000 SERIES MXX
            //*****************************************************************************
            //*****************************************************************************
            { 0x1d10, {"NVIDIA GeForce MX150 GP108M", 3, 65536, 49152, 64} },

            { 0x1d12, {"NVIDIA GeForce MX150 GP108M", 3, 65536, 49152, 64} },

            { 0x1c90, {"NVIDIA GeForce MX150 GP107M", 3, 65536, 49152, 64} },

            { 0x1d11, {"NVIDIA GeForce MX230", 2, 65536, 49152, 64} },

            { 0x1d13, {"NVIDIA GeForce MX250 GP108M", 3, 65536, 49152, 64} },

            { 0x1d52, {"NVIDIA GeForce MX250 GP108BM", 3, 65536, 49152, 64} },

            { 0x1d16, {"NVIDIA GeForce MX330 GP108M", 3, 65536, 49152, 64} },

            { 0x1d56, {"NVIDIA GeForce MX350 GP108BM", 5, 65536, 49152, 64} },

            { 0x1c94, {"NVIDIA GeForce MX350", 5, 65536, 49152, 64} },

            { 0x1c96, {"NVIDIA GeForce MX350", 5, 65536, 49152, 64} },

            //*****************************************************************************
            //*****************************************************************************
            //1660 SERIES NON MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x1f83, {"NVIDIA GeForce GTX 1630", 8, 65536, 65536, 32} },

            { 0x1f82, {"NVIDIA GeForce GTX 1650 TU117", 14, 65536, 65536, 32} },

            { 0x2188, {"NVIDIA GeForce GTX 1650 TU116", 14, 65536, 65536, 32} },

            { 0x1f0a, {"NVIDIA GeForce GTX 1650 TU106", 14, 65536, 65536, 32} },

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1650 GDDR6", 14, 65536, 65536, 32} },

            { 0x2187, {"NVIDIA GeForce GTX 1650 SUPER", 20, 65536, 65536, 32} },

            { 0x2184, {"NVIDIA GeForce GTX 1660", 22, 65536, 65536, 32} },

            { 0x1f09, {"NVIDIA GeForce GTX 1660 SUPER TU106", 22, 65536, 65536, 32} },

            { 0x21c4, {"NVIDIA GeForce GTX 1660 SUPER TU116", 22, 65536, 65536, 32} },

            { 0x2182, {"NVIDIA GeForce GTX 1660 Ti", 24, 65536, 65536, 32} },

            //*****************************************************************************
            //*****************************************************************************
            //1660 SERIES MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x1f91, {"NVIDIA GeForce GTX 1650 Max-Q", 16, 65536, 65536, 32} },

            { 0x1f96, {"NVIDIA GeForce GTX 1650 Max-Q", 16, 65536, 65536, 32} },

            { 0x1f99, {"NVIDIA GeForce GTX 1650 Max-Q", 16, 65536, 65536, 32} },

            { 0x1f9d, {"NVIDIA GeForce GTX 1650 Max-Q", 16, 65536, 65536, 32} },

            { 0x1f92, {"NVIDIA GeForce GTX 1650 Mobile", 16, 65536, 65536, 32} },

            { 0x1f94, {"NVIDIA GeForce GTX 1650 Mobile", 16, 65536, 65536, 32} },

            { 0x1fd9, {"NVIDIA GeForce GTX 1650 Mobile Refresh", 16, 65536, 65536, 32} },

            { 0x1fdd, {"NVIDIA GeForce GTX 1650 Mobile Refresh", 16, 65536, 65536, 32} },

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1650 Ti Max-Q", 16, 65536, 65536, 32} },

            { 0x1f95, {"NVIDIA GeForce GTX 1650 Ti Mobile TU117M", 16, 65536, 65536, 32} },

            { 0x2192, {"NVIDIA GeForce GTX 1650 Ti Mobile TU116M", 16, 65536, 65536, 32} },

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce GTX 1660 Ti Max-Q", 24, 65536, 65536, 32} },

            { 0x2191, {"NVIDIA GeForce GTX 1660 Ti Mobile TU116M", 24, 65536, 65536, 32} },

            { 0x21d1, {"NVIDIA GeForce GTX 1660 Ti Mobile TU116BM", 24, 65536, 65536, 32} },

            //*****************************************************************************
            //*****************************************************************************
            //2000 SERIES
            //*****************************************************************************
            //*****************************************************************************
            { 0x1f08, {"NVIDIA GeForce RTX 2060 Rev. A", 30, 65536, 65536, 32} },

            { 0x1e89, {"NVIDIA GeForce RTX 2060 TU104", 30, 65536, 65536, 32} },

            { 0x1f03, {"NVIDIA GeForce RTX 2060 12 GB", 34, 65536, 65536, 32} },

            { 0x1f06, {"NVIDIA GeForce RTX 2060 SUPER", 34, 65536, 65536, 32} },

            { 0x1f42, {"NVIDIA GeForce RTX 2060 SUPER", 34, 65536, 65536, 32} },

            { 0x1f47, {"NVIDIA GeForce RTX 2060 SUPER", 34, 65536, 65536, 32} },

            { 0x1f02, {"NVIDIA GeForce RTX 2070", 36, 65536, 65536, 32} },

            { 0x1f07, {"NVIDIA GeForce RTX 2070 Rev. A", 36, 65536, 65536, 32} },

            { 0x1e84, {"NVIDIA GeForce RTX 2070 SUPER", 40, 65536, 65536, 32} },

            { 0x1ec2, {"NVIDIA GeForce RTX 2070 SUPER", 40, 65536, 65536, 32} },

            { 0x1ec7, {"NVIDIA GeForce RTX 2070 SUPER", 40, 65536, 65536, 32} },

            { 0x1e82, {"NVIDIA GeForce RTX 2080", 46, 65536, 65536, 32} },

            { 0x1e87, {"NVIDIA GeForce RTX 2080 Rev. A", 46, 65536, 65536, 32} },

            { 0x1e81, {"NVIDIA GeForce RTX 2080 SUPER", 48, 65536, 65536, 32} },

            { 0x1e04, {"NVIDIA GeForce RTX 2080 Ti", 68, 65536, 65536, 32} },

            { 0x1e07, {"NVIDIA GeForce RTX 2080 Ti Rev. A", 68, 65536, 65536, 32} },

            { 0x1e02, {"NVIDIA TITAN RTX", 72, 65536, 65536, 32} },

            //*****************************************************************************
            //*****************************************************************************
            //2000 SERIES MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x1f12, {"NVIDIA GeForce RTX 2060 Max-Q", 30, 65536, 65536, 32} },

            { 0x1f11, {"NVIDIA GeForce RTX 2060 Mobile TU106M", 30, 65536, 65536, 32} },

            { 0x1f15, {"NVIDIA GeForce RTX 2060 Mobile TU106M", 30, 65536, 65536, 32} },

            { 0x1f51, {"NVIDIA GeForce RTX 2060 Mobile TU106BM", 30, 65536, 65536, 32} },

            { 0x1f55, {"NVIDIA GeForce RTX 2060 Mobile TU106BM", 30, 65536, 65536, 32} },

            //Not found, likely duplicate
            //{ 0x, {"NVIDIA GeForce RTX 2060 Mobile Refresh", 30, 65536, 65536, 32} },

            //{ 0x, {"NVIDIA GeForce RTX 2060 Max-Q Refresh", 30, 65536, 65536, 32} },

            //Not found!
            //{ 0x, {"NVIDIA GeForce RTX 2060 SUPER Mobile", 34, 65536, 65536, 32} },

            { 0x1f10, {"NVIDIA GeForce RTX 2070 Mobile", 36, 65536, 65536, 32} },

            { 0x1f54, {"NVIDIA GeForce RTX 2070 Mobile Refresh", 36, 65536, 65536, 32} },

            { 0x1f50, {"NVIDIA GeForce RTX 2070 Max-Q", 36, 65536, 65536, 32} },

            { 0x1f14, {"NVIDIA GeForce RTX 2070 Max-Q Refresh", 36, 65536, 65536, 32} },

            { 0x1e91, {"NVIDIA GeForce RTX 2070 SUPER Max-Q", 40, 65536, 65536, 32} },

            { 0x1ed1, {"NVIDIA GeForce RTX 2070 SUPER Mobile", 40, 65536, 65536, 32} },

            { 0x1e90, {"NVIDIA GeForce RTX 2080 Mobile", 46, 65536, 65536, 32} },

            { 0x1ed0, {"NVIDIA GeForce RTX 2080 Max-Q", 46, 65536, 65536, 32} },

            { 0x1e93, {"NVIDIA GeForce RTX 2080 SUPER Mobile", 48, 65536, 65536, 32} },

            { 0x1ed3, {"NVIDIA GeForce RTX 2080 SUPER Max-Q", 48, 65536, 65536, 32} },

            //*****************************************************************************
            //*****************************************************************************
            //2000 SERIES MXXX
            //*****************************************************************************
            //*****************************************************************************
            { 0x1f97, {"NVIDIA GeForce MX450 12W", 14, 65536, 65536, 32} },

            { 0x1f98, {"NVIDIA GeForce MX450 30.5W 8Gbps", 14, 65536, 65536, 32} },

            { 0x1f9c, {"NVIDIA GeForce MX450 30.5W 10Gbps", 14, 65536, 65536, 32} },

            { 0x1f9f, {"NVIDIA GeForce MX550", 16, 65536, 131072, 32} },

            { 0x1fa0, {"NVIDIA GeForce MX550", 16, 65536, 131072, 32} },

            //*****************************************************************************
            //*****************************************************************************
            //3000 SERIES NON MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x2583, {"NVIDIA GeForce RTX 3050 4 GB", 16, 65536, 131072, 48} },

            { 0x2584, {"NVIDIA GeForce RTX 3050 6 GB", 18, 65536, 131072, 48} },

            { 0x2508, {"NVIDIA GeForce RTX 3050 OEM", 18, 65536, 131072, 48} },

            { 0x2582, {"NVIDIA GeForce RTX 3050 8 GB GA107", 20, 65536, 131072, 48} },

            { 0x2507, {"NVIDIA GeForce RTX 3050 8 GB GA106", 20, 65536, 131072, 48} },

            { 0x2501, {"NVIDIA GeForce RTX 3060 8 GB", 28, 65536, 131072, 48} },

            { 0x2503, {"NVIDIA GeForce RTX 3060 8 GB", 28, 65536, 131072, 48} },

            { 0x2504, {"NVIDIA GeForce RTX 3060 Lite Hash Rate 8 GB", 28, 65536, 131072, 48} },

            { 0x24c7, {"NVIDIA GeForce RTX 3060 8 GB GA104", 28, 65536, 131072, 48} },

            { 0x2487, {"NVIDIA GeForce RTX 3060 12 GB GA104", 28, 65536, 131072, 48} },

            { 0x2544, {"NVIDIA GeForce RTX 3060 12 GB GA106", 28, 65536, 131072, 48} },

            { 0x2509, {"NVIDIA GeForce RTX 3060 12GB Rev. 2 GA106", 28, 65536, 131072, 48} },

            //Not Found, rare card?
            //{ 0x, {"NVIDIA GeForce RTX 3060 3840SP", 30, 65536, 131072, 48} },

            { 0x2414, {"NVIDIA GeForce RTX 3060 Ti GA103", 38, 65536, 131072, 48} },

            { 0x2486, {"NVIDIA GeForce RTX 3060 Ti GA104", 38, 65536, 131072, 48} },

            { 0x2489, {"NVIDIA GeForce RTX 3060 Ti Lite Hash Rate GA104", 38, 65536, 131072, 48} },

            { 0x24c9, {"NVIDIA GeForce RTX 3060 Ti GDDR6X", 38, 65536, 131072, 48} },

            { 0x2484, {"NVIDIA GeForce RTX 3070", 46, 65536, 131072, 48} },

            { 0x24c8, {"NVIDIA GeForce RTX 3070 GDDR6X", 46, 65536, 131072, 48} },

            { 0x2488, {"NVIDIA GeForce RTX 3070 Lite Hash Rate", 46, 65536, 131072, 48} },

            { 0x2482, {"NVIDIA GeForce RTX 3070 Ti GA104", 48, 65536, 131072, 48} },

            { 0x2207, {"NVIDIA GeForce RTX 3070 Ti 8 GB GA102", 48, 65536, 131072, 48} },

            { 0x2206, {"NVIDIA GeForce RTX 3080", 70, 65536, 131072, 48} },

            { 0x220a, {"NVIDIA GeForce RTX 3080 12 GB", 70, 65536, 131072, 48} },

            { 0x2216, {"NVIDIA GeForce RTX 3080 Lite Hash Rate", 70, 65536, 131072, 48} },

            { 0x2208, {"NVIDIA GeForce RTX 3080 Ti", 80, 65536, 131072, 48} },

            { 0x2205, {"NVIDIA GeForce RTX 3080 Ti 20 GB", 80, 65536, 131072, 48} },

            { 0x2204, {"NVIDIA GeForce RTX 3090", 82, 65536, 131072, 48} },

            { 0x2203, {"NVIDIA GeForce RTX 3090 Ti", 84, 65536, 131072, 48} },

            //*****************************************************************************
            //*****************************************************************************
            //3000 SERIES MOBILE
            //*****************************************************************************
            //*****************************************************************************
            //2050 but GA107
            { 0x25ad, {"NVIDIA GeForce RTX 2050 Mobile / Max-Q GA107", 16, 65536, 65536, 48} },

            { 0x25ed, {"NVIDIA GeForce RTX 2050 Mobile / Max-Q GA107", 16, 65536, 65536, 48} },

            { 0x25a9, {"NVIDIA GeForce RTX 2050 Mobile / Max-Q GA107M", 16, 65536, 65536, 48} },

            { 0x25a2, {"NVIDIA GeForce RTX 3050 Mobile GA107M", 16, 65536, 131072, 48} },

            { 0x25a5, {"NVIDIA GeForce RTX 3050 Max-Q GA107M", 16, 65536, 131072, 48} },

            { 0x25ab, {"NVIDIA GeForce RTX 3050 4GB Laptop GPU GA107M", 16, 65536, 131072, 48} },

            { 0x25e2, {"NVIDIA GeForce RTX 3050 Mobile GA107BM", 20, 65536, 131072, 48} },

            { 0x25e5, {"NVIDIA GeForce RTX 3050 Mobile GA107BM", 20, 65536, 131072, 48} },

            { 0x25ac, {"NVIDIA GeForce RTX 3050 6GB Laptop GPU GN20-P0-R-K2", 20, 65536, 131072, 48} },

            { 0x25ec, {"NVIDIA GeForce RTX 3050 6GB Laptop GPU GN20-P0-R-K2", 20, 65536, 131072, 48} },

            { 0x2523, {"NVIDIA GeForce RTX 3050 Ti Mobile / Max-Q GA106M", 20, 65536, 131072, 48} },

            { 0x2563, {"NVIDIA GeForce RTX 3050 Ti Mobile / Max-Q GA106M", 20, 65536, 131072, 48} },

            { 0x25a0, {"NVIDIA GeForce RTX 3050 Ti Mobile GA107M", 20, 65536, 131072, 48} },

            { 0x25e0, {"NVIDIA GeForce RTX 3050 Ti Mobile GA107BM", 20, 65536, 131072, 48} },

            { 0x2520, {"NVIDIA GeForce RTX 3060 Max-Q", 30, 65536, 131072, 48} },

            { 0x2521, {"NVIDIA GeForce RTX 3060 Mobile", 30, 65536, 131072, 48} },

            { 0x2560, {"NVIDIA GeForce RTX 3060 Max-Q", 30, 65536, 131072, 48} },

            { 0x2561, {"NVIDIA GeForce RTX 3060 Mobile", 30, 65536, 131072, 48} },

            { 0x249d, {"NVIDIA GeForce RTX 3070 Mobile / Max-Q", 40, 65536, 131072, 48} },

            { 0x24dd, {"NVIDIA GeForce RTX 3070 Mobile / Max-Q", 40, 65536, 131072, 48} },

            { 0x24a0, {"NVIDIA GeForce RTX 3070 Ti Mobile / Max-Q", 46, 65536, 131072, 48} },

            { 0x24e0, {"NVIDIA GeForce RTX 3070 Ti Mobile / Max-Q", 46, 65536, 131072, 48} },

            { 0x249c, {"NVIDIA GeForce RTX 3080 Mobile / Max-Q 8GB/16GB", 48, 65536, 131072, 48} },

            { 0x24dc, {"NVIDIA GeForce RTX 3080 Mobile / Max-Q 8GB/16GB", 48, 65536, 131072, 48} },

            { 0x2420, {"NVIDIA GeForce RTX 3080 Ti Mobile / Max-Q", 58, 65536, 131072, 48} },

            { 0x2460, {"NVIDIA GeForce RTX 3080 Ti Mobile / Max-Q", 58, 65536, 131072, 48} },

            //*****************************************************************************
            //*****************************************************************************
            //3000 SERIES MXX
            //*****************************************************************************
            //*****************************************************************************
            { 0x25a6, {"NVIDIA GeForce MX570", 16, 65536, 131072, 48} },

            { 0x25a7, {"NVIDIA GeForce MX570", 16, 65536, 131072, 48} },

            { 0x25aa, {"NVIDIA GeForce MX570 A", 16, 65536, 131072, 48} },

            //*****************************************************************************
            //*****************************************************************************
            //4000 SERIES NON MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x2882, {"NVIDIA GeForce RTX 4060", 24, 65536, 131072, 48} },

            { 0x2803, {"NVIDIA GeForce RTX 4060 Ti 8 GB", 34, 65536, 131072, 48} },

            { 0x2805, {"NVIDIA GeForce RTX 4060 Ti 16 GB", 34, 65536, 131072, 48} },

            { 0x2786, {"NVIDIA GeForce RTX 4070", 46, 65536, 131072, 48} },

            { 0x2783, {"NVIDIA GeForce RTX 4070 SUPER", 56, 65536, 131072, 48} },

            { 0x2782, {"NVIDIA GeForce RTX 4070 Ti", 60, 65536, 131072, 48} },

            { 0x2705, {"NVIDIA GeForce RTX 4070 Ti SUPER", 66, 65536, 131072, 48} },

            { 0x2704, {"NVIDIA GeForce RTX 4080", 76, 65536, 131072, 48} },

            { 0x2702, {"NVIDIA GeForce RTX 4080 SUPER", 80, 65536, 131072, 48} },

            { 0x2703, {"NVIDIA GeForce RTX 4080 SUPER", 80, 65536, 131072, 48} },

            { 0x2684, {"NVIDIA GeForce RTX 4090", 128, 65536, 131072, 48} },

            { 0x2685, {"NVIDIA GeForce RTX 4090 D", 114, 65536, 131072, 48} },

            //*****************************************************************************
            //*****************************************************************************
            //4000 SERIES MOBILE
            //*****************************************************************************
            //*****************************************************************************
            { 0x28a1, {"NVIDIA GeForce RTX 4050 Max-Q / Mobile", 20, 65536, 131072, 48} },

            { 0x28e1, {"NVIDIA GeForce RTX 4050 Max-Q / Mobile", 20, 65536, 131072, 48} },

            { 0x28a0, {"NVIDIA GeForce RTX 4060 Max-Q / Mobile", 24, 65536, 131072, 48} },

            { 0x28e0, {"NVIDIA GeForce RTX 4060 Max-Q / Mobile", 24, 65536, 131072, 48} },

            { 0x2820, {"NVIDIA GeForce RTX 4070 Max-Q / Mobile", 36, 65536, 131072, 48} },

            { 0x2860, {"NVIDIA GeForce RTX 4070 Max-Q / Mobile", 36, 65536, 131072, 48} },

            { 0x27a0, {"NVIDIA GeForce RTX 4080 Max-Q / Mobile", 58, 65536, 131072, 48} },

            { 0x27e0, {"NVIDIA GeForce RTX 4080 Max-Q / Mobile", 58, 65536, 131072, 48} },

            { 0x2717, {"NVIDIA GeForce RTX 4090 Mobile", 76, 65536, 131072, 48} },

            //Not found ?
            //{ 0x, {"NVIDIA GeForce RTX 4090 Max-Q", 76, 65536, 131072, 48} },
        };

        return adapterTable;
    }

    static std::unordered_map<uint32_t, adapterInfoAmd> InitializeAMDTuningTable()
    {
        //Unclear if Pro overlaps with RX, so included
        std::unordered_map<uint32_t, adapterInfoAmd> adapterTable
        {
            //*****************************************************************************
            //*****************************************************************************
            //NAVI 10 
            //*****************************************************************************
            //*****************************************************************************
            { 0x7310, {"Radeon Pro W5700X"}},

            { 0x7312, {"Radeon Pro W5700"}},

            { 0x7319, {"Radeon Pro 5700 XT"}},

            { 0x731b, {"Radeon Pro 5700"}},

            { 0x731f, {"Radeon RX 5600 OEM/5600 XT / 5700/5700 XT"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 14 
            //*****************************************************************************
            //*****************************************************************************
            { 0x734f, {"Radeon Pro W5300M"}},

            { 0x7341, {"Radeon Pro W5500"}},

            { 0x7347, {"Radeon Pro W5500M"}},

            //This is also 5300 apparently
            { 0x7340, {"Radeon RX 5500/5500M / Pro 5500M"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 21 
            //*****************************************************************************
            //*****************************************************************************
            { 0x73a1, {"Radeon Pro V620"}},

            { 0x73a2, {"Radeon Pro W6900X"}},

            { 0x73a3, {"Radeon Pro W6800"}},

            { 0x73a5, {"Radeon RX 6950 XT"}},

            { 0x73ab, {"Radeon Pro W6800X/Radeon Pro W6800X Duo"}},

            { 0x73ae, {"Radeon Pro V620 MxGPU"}},

            { 0x73af, {"Radeon RX 6900 XT"}},

            { 0x73bf, {"Radeon RX 6800/6800 XT / 6900 XT"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 22 
            //*****************************************************************************
            //*****************************************************************************
            { 0x73df, {"Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 23 
            //*****************************************************************************
            //*****************************************************************************
            { 0x73e1, {"Radeon PRO W6600M"}},

            { 0x73e3, {"Radeon PRO W6600"}},

            { 0x73ef, {"Radeon RX 6650 XT / 6700S / 6800S"}},

            { 0x73ff, {"Radeon RX 6600/6600 XT/6600M"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 24 
            //*****************************************************************************
            //*****************************************************************************
            { 0x7421, {"Radeon PRO W6500M"}},

            { 0x7422, {"Radeon PRO W6400"}},

            { 0x7423, {"Radeon PRO W6300/W6300M"}},

            { 0x7424, {"Radeon RX 6300"}},

            { 0x743f, {"Radeon RX 6400/6500 XT/6500M"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 31 
            //*****************************************************************************
            //*****************************************************************************
            { 0x7448, {"Radeon Pro W7900"}},

            { 0x744c, {"Radeon RX 7900 XT/7900 XTX/7900M"}},

            { 0x745e, {"Radeon Pro W7800"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 32 
            //*****************************************************************************
            //*****************************************************************************
            { 0x7470, {"Radeon PRO W7700"}},

            { 0x747e, {"Radeon RX 7700 XT / 7800 XT"}},

            //*****************************************************************************
            //*****************************************************************************
            //NAVI 33 
            //*****************************************************************************
            //*****************************************************************************
            { 0x73f0, {"Radeon RX 7600M XT"}},

            { 0x7480, {"Radeon RX 7700S/7600/7600S/7600M XT/PRO W7600"}},

            { 0x7483, {"Radeon RX 7600M/7600M XT"} },

            { 0x7489, {"Radeon Pro W7500"} },
        };

        return adapterTable;
    }

    static inline uint32_t max(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    static inline uint32_t partitionSize(
        uint32_t keysPerThread,
        uint32_t threadsPerThreadBlock)
    {
        return keysPerThread * threadsPerThreadBlock;
    }

    static inline uint32_t histogramSharedMemory(
        uint32_t threadsPerThreadBlock,
        uint32_t simdWidth)
    {
        const uint32_t radix = 256;
        return threadsPerThreadBlock / simdWidth * radix;
    }

    static inline uint32_t combinedPartitionSharedMemory(
        uint32_t partitionSize)
    {
        const uint32_t radix = 256;
        return partitionSize + radix;
    }

    static inline GPUSorting::TuningParameters GetGenericTuningParameters(
        const GPUSorting::DeviceInfo& devInfo)
    {
        bool genericShouldLock = false;
        uint32_t genericKeysPerThread = 7;
        uint32_t genericThreadsPerThreadBlock = 256;
        uint32_t genericPartSize =
            partitionSize(genericKeysPerThread, genericThreadsPerThreadBlock);
        uint32_t genericCombinedPartitionSharedMemory =
            combinedPartitionSharedMemory(genericPartSize);
        uint32_t genericHistogramSharedMemory =
            histogramSharedMemory(genericThreadsPerThreadBlock, devInfo.SIMDWidth);
        uint32_t genericTotalSharedMem = 4096;

        return {
            genericShouldLock,
            genericKeysPerThread,
            genericThreadsPerThreadBlock,
            genericPartSize,
            genericTotalSharedMem };
    }

    static inline GPUSorting::TuningParameters calcKeysTuningParametersNvidia(
        const GPUSorting::DeviceInfo& devInfo,
        const adapterInfoNvidia& info)
    {
        const uint32_t lanesPerWarp = 32;

        GPUSorting::TuningParameters tuningParams;
        uint32_t keysPerThread = 0;
        uint32_t threadsPerThreadBlock = 0;
        bool shouldLock = false;

        //In the future we may have to take varied register file sizes
        //into account, but for now all register files are the same
        switch (info.sharedMemoryPerSM)
        {
        case 16384:
            keysPerThread = 7;
            threadsPerThreadBlock = 512; //3584
            break;
        case 49152:
            keysPerThread = 7;
            threadsPerThreadBlock = 512; //3584
            break;
        case 65536:
            keysPerThread = 15;
            threadsPerThreadBlock = 256; //3840
            break;
        case 131072:
            keysPerThread = 15;
            threadsPerThreadBlock = 256; //3840
            break;
        default:
            return GetGenericTuningParameters(devInfo);
        }

        uint32_t partitionSize = keysPerThread * threadsPerThreadBlock;
        uint32_t histSharedMemory =
            histogramSharedMemory(threadsPerThreadBlock, lanesPerWarp);
        uint32_t combinedPartSize =
            combinedPartitionSharedMemory(partitionSize);
        uint32_t totalSharedMemory = max(histSharedMemory, combinedPartSize);

        tuningParams = {
            shouldLock,
            keysPerThread,
            threadsPerThreadBlock,
            partitionSize,
            totalSharedMemory };

        return tuningParams;
    }

    static inline GPUSorting::TuningParameters calcPairsTuningParametersNvidia(
        const GPUSorting::DeviceInfo& devInfo,
        const adapterInfoNvidia& info)
    {
        const uint32_t lanesPerWarp = 32;

        GPUSorting::TuningParameters tuningParams;
        uint32_t keysPerThread = 0;
        uint32_t threadsPerThreadBlock = 0;
        bool shouldLock = false;

        switch (info.sharedMemoryPerSM)
        {
        case 16384:
            keysPerThread = 7;
            threadsPerThreadBlock = 512; //3584
            break;
        case 49152:
            keysPerThread = 15;
            threadsPerThreadBlock = 512; //7680
            break;
        case 65536:
            keysPerThread = 15;
            threadsPerThreadBlock = 512; //7680
            break;
        case 131072:
            keysPerThread = 15;
            threadsPerThreadBlock = 512; //7680
            break;
        default:
            return GetGenericTuningParameters(devInfo);
        }

        uint32_t partitionSize = keysPerThread * threadsPerThreadBlock;
        uint32_t histSharedMemory =
            histogramSharedMemory(threadsPerThreadBlock, lanesPerWarp);
        uint32_t combinedPartSize =
            combinedPartitionSharedMemory(partitionSize);
        uint32_t totalSharedMemory = max(histSharedMemory, combinedPartSize);

        tuningParams = {
            shouldLock,
            keysPerThread,
            threadsPerThreadBlock,
            partitionSize,
            totalSharedMemory };

        return tuningParams;

    }

    //RDNA very straightforward, as there appears to be no 
    //VGPR or LDS variation between generations.
    static inline GPUSorting::TuningParameters calcKeysTuningParametersRDNA()
    {
        //We lock the wave size to 32 because we want want WGPs not CUs
        bool shouldLockToW32 = true;
        const uint32_t lanesPerWave = 32;

        uint32_t keysPerThread = 7;
        uint32_t threadsPerThreadBlock = 512;	
        uint32_t partSize = partitionSize(keysPerThread, threadsPerThreadBlock); //3584
        uint32_t histSharedMemory = histogramSharedMemory(threadsPerThreadBlock, lanesPerWave);
        uint32_t combinedPartSize = 
            combinedPartitionSharedMemory(partSize);
        uint32_t totalSharedMemory = 
            max(histSharedMemory, combinedPartSize);

        const GPUSorting::TuningParameters tuningParams = {
            shouldLockToW32,
            keysPerThread,
            threadsPerThreadBlock,
            partSize,
            totalSharedMemory };

        return tuningParams;
    }

    static inline GPUSorting::TuningParameters calcPairsTuningParametersRDNA()
    {
        //We lock the wave size to 32 because we want want WGPs not CUs
        const bool shouldLockToW32 = true;
        const uint32_t lanesPerWave = 32;

        uint32_t keysPerThread = 5;
        uint32_t threadsPerThreadBlock = 512;
        uint32_t partSize = partitionSize(keysPerThread, threadsPerThreadBlock); //2560
        uint32_t histSharedMemory = histogramSharedMemory(threadsPerThreadBlock, lanesPerWave);
        uint32_t combinedPartSize =
            combinedPartitionSharedMemory(partSize);
        uint32_t totalSharedMemory =
            max(histSharedMemory, combinedPartSize);

        const GPUSorting::TuningParameters tuningParams = {
            shouldLockToW32,
            keysPerThread,
            threadsPerThreadBlock,
            partSize,
            totalSharedMemory };

        return tuningParams;
    }

    static GPUSorting::TuningParameters CalculateTuningParametersNvidia(
        const GPUSorting::DeviceInfo& devInfo,
        const GPUSorting::MODE& gpuSortMode)
    {
        GPUSorting::TuningParameters tuningParams;
        std::unordered_map<uint32_t, TunerHelper::adapterInfoNvidia> table =
            TunerHelper::InitializeNvidiaTuningTable();
        auto result = table.find(devInfo.deviceId);
        if (result != table.end())
        {
            tuningParams = gpuSortMode == GPUSorting::MODE_KEYS_ONLY ?
                TunerHelper::calcKeysTuningParametersNvidia(devInfo, result->second) :
                TunerHelper::calcPairsTuningParametersNvidia(devInfo, result->second);
        }
        else
        {
#ifdef _DEBUG
            printf("Device not found in tuning table, reverting to generic tuning preset.");
#endif
            tuningParams = TunerHelper::GetGenericTuningParameters(devInfo);
        }

        return tuningParams;
    }

    static GPUSorting::TuningParameters CalculateTuningParametersAMD(
        const GPUSorting::DeviceInfo& devInfo,
        const GPUSorting::MODE& gpuSortMode)
    {
        GPUSorting::TuningParameters tuningParams;
        std::unordered_map<uint32_t, TunerHelper::adapterInfoAmd> table =
            TunerHelper::InitializeAMDTuningTable();
        auto result = table.find(devInfo.deviceId);
        if (result != table.end())
        {
            tuningParams = gpuSortMode == GPUSorting::MODE_KEYS_ONLY ?
                TunerHelper::calcKeysTuningParametersRDNA() :
                TunerHelper::calcPairsTuningParametersRDNA();
        }
        else
        {
#ifdef _DEBUG
            printf("Device not found in tuning table, reverting to generic tuning preset.");
#endif
            tuningParams = TunerHelper::GetGenericTuningParameters(devInfo);
        }

        return tuningParams;
    }
}

namespace Tuner
{
    static inline GPUSorting::TuningParameters GetTuningParameters(
        const GPUSorting::DeviceInfo& devInfo,
        const GPUSorting::MODE& gpuSortMode)
    {
        GPUSorting::TuningParameters tuningParams;
        switch (devInfo.vendorId)
        {
        case 0x1002:
            tuningParams = TunerHelper::CalculateTuningParametersAMD(devInfo, gpuSortMode);
            break;
        case 0x10de:
            tuningParams = TunerHelper::CalculateTuningParametersNvidia(devInfo, gpuSortMode);
            break;
        default:
#ifdef _DEBUG
            printf("No tuning preset for vendor, reverting to generic tuning preset.");
#endif
            tuningParams = TunerHelper::GetGenericTuningParameters(devInfo);
            break;
        }

#ifdef _DEBUG
        printf("\nTuning Parameters: \n");
        printf("ShouldLockWaves: %u\n", tuningParams.shouldLockWavesTo32);
        printf("KeysPerThread: %u\n", tuningParams.keysPerThread);
        printf("ThreadsPerThreadBlock: %u\n", tuningParams.threadsPerThreadblock);
        printf("PartitionSize: %u\n", tuningParams.partitionSize);
        printf("TotalSharedMemory: %u\n\n", tuningParams.totalSharedMemory);
#endif

        return tuningParams;
    }
}