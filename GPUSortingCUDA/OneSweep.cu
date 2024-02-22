/******************************************************************************
 * GPUSorting
 * OneSweep Implementation
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 * 
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 ******************************************************************************/
#include "OneSweep.cuh"

#define RADIX               256     //Number of digit bins
#define RADIX_MASK          255     //Mask of digit bins, to extract digits
#define RADIX_LOG           8       //log2(RADIX)

#define SEC_RADIX_START     256
#define THIRD_RADIX_START   512
#define FOURTH_RADIX_START  768

//For the upfront global histogram kernel
#define G_HIST_PART_SIZE	65536
#define G_HIST_VEC_SIZE		16384

//For the digit binning
#define BIN_PART_SIZE       7680                                    //Partition tile size in k_DigitBinning
#define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
#define BIN_SUB_PART_SIZE   480                                     //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_WARPS           16                                      //Warps per threadblock in k_DigitBinning
#define BIN_KEYS_PER_THREAD 15                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)        //Starting offset of a partition tile

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0                                       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1                                       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2                                       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3                                       //Mask used to retrieve flag values

__global__ void OneSweep::GlobalHistogram(
    uint32_t* sort,
    uint32_t* globalHistogram,
    uint32_t size)
{
    __shared__ uint32_t s_globalHistFirst[RADIX * 2];
    __shared__ uint32_t s_globalHistSec[RADIX * 2];
    __shared__ uint32_t s_globalHistThird[RADIX * 2];
    __shared__ uint32_t s_globalHistFourth[RADIX * 2];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
    {
        s_globalHistFirst[i] = 0;
        s_globalHistSec[i] = 0;
        s_globalHistThird[i] = 0;
        s_globalHistFourth[i] = 0;
    }
    __syncthreads();
    
    //histogram
    {
        //64 threads : 1 histogram in shared memory
        uint32_t* s_wavesHistFirst = &s_globalHistFirst[threadIdx.x / 64 * RADIX];
        uint32_t* s_wavesHistSec = &s_globalHistSec[threadIdx.x / 64 * RADIX];
        uint32_t* s_wavesHistThird = &s_globalHistThird[threadIdx.x / 64 * RADIX];
        uint32_t* s_wavesHistFourth = &s_globalHistFourth[threadIdx.x / 64 * RADIX];

        if (blockIdx.x < gridDim.x - 1)
        {
            const uint32_t partEnd = (blockIdx.x + 1) * G_HIST_VEC_SIZE;
            for (uint32_t i = threadIdx.x + (blockIdx.x * G_HIST_VEC_SIZE); i < partEnd; i += blockDim.x)
            {
                uint4 t[1] = { reinterpret_cast<uint4*>(sort)[i] };

                atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
                atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[1]], 1);
                atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[2]], 1);
                atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[3]], 1);

                atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[4]], 1);
                atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[5]], 1);
                atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[6]], 1);
                atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[7]], 1);

                atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[8]], 1);
                atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[9]], 1);
                atomicAdd(&s_globalHistThird[reinterpret_cast<uint8_t*>(t)[10]], 1);
                atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[11]], 1);

                atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[12]], 1);
                atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[13]], 1);
                atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[14]], 1);
                atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[15]], 1);
            }
        }

        if (blockIdx.x == gridDim.x - 1)
        {
            for (uint32_t i = threadIdx.x + (blockIdx.x * G_HIST_PART_SIZE); i < size; i += blockDim.x)
            {
                uint32_t t[1] = { sort[i] };
                atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
                atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[1]], 1);
                atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[2]], 1);
                atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[3]], 1);
            }
        }
    }
    __syncthreads();

    //reduce and add to device
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
    {
        atomicAdd(&globalHistogram[i], s_globalHistFirst[i] + s_globalHistFirst[i + RADIX]);
        atomicAdd(&globalHistogram[i + SEC_RADIX_START], s_globalHistSec[i] + s_globalHistSec[i + RADIX]);
        atomicAdd(&globalHistogram[i + THIRD_RADIX_START], s_globalHistThird[i] + s_globalHistThird[i + RADIX]);
        atomicAdd(&globalHistogram[i + FOURTH_RADIX_START], s_globalHistFourth[i] + s_globalHistFourth[i + RADIX]);
    }
}

__global__ void OneSweep::Scan(
    uint32_t* globalHistogram,
    uint32_t* firstPassHistogram,
    uint32_t* secPassHistogram,
    uint32_t* thirdPassHistogram,
    uint32_t* fourthPassHistogram)
{
    __shared__ uint32_t s_scan[RADIX];

    s_scan[threadIdx.x] = InclusiveWarpScanCircularShift(globalHistogram[threadIdx.x + blockIdx.x * RADIX]);
    __syncthreads();

    if(threadIdx.x < (RADIX >> LANE_LOG))
        s_scan[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_scan[threadIdx.x << LANE_LOG]);
    __syncthreads();

    switch (blockIdx.x)
    {
    case 0:
        firstPassHistogram[threadIdx.x] = 
            (s_scan[threadIdx.x] + (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1) : 0)) << 2 | FLAG_INCLUSIVE;
        break;
    case 1:
        secPassHistogram[threadIdx.x] =
            (s_scan[threadIdx.x] + (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1) : 0)) << 2 | FLAG_INCLUSIVE;
        break;
    case 2:
        thirdPassHistogram[threadIdx.x] =
            (s_scan[threadIdx.x] + (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1) : 0)) << 2 | FLAG_INCLUSIVE;
        break;
    case 3:
        fourthPassHistogram[threadIdx.x] =
            (s_scan[threadIdx.x] + (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1) : 0)) << 2 | FLAG_INCLUSIVE;
        break;
    default:
        break;
    }
}

__global__ void OneSweep::DigitBinningPass(
    uint32_t* sort,
    uint32_t* alt,
    volatile uint32_t* passHistogram,
    volatile uint32_t* index,
    uint32_t size,
    uint32_t radixShift)
{
    __shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
    __shared__ uint32_t s_localHistogram[RADIX];
    volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)  //unnecessary work for last partion but still a win to avoid another barrier
        s_warpHistograms[i] = 0;

    //atomically assign partition tiles
    if (threadIdx.x == 0)
        s_warpHistograms[BIN_PART_SIZE - 1] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
    __syncthreads();
    const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];

    //To handle input sizes not perfect multiples of the partition tile size
    if (partitionIndex < gridDim.x - 1)
    {
        //load keys
        uint32_t keys[BIN_KEYS_PER_THREAD];
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = sort[t];

        uint16_t offsets[BIN_KEYS_PER_THREAD];

        //WLMS
        #pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            //CUB version "match any
            /*
            unsigned warpFlags;
            #pragma unroll
            for (int k = 0; k < RADIX_LOG; ++k)
            {
                uint32_t mask;
                uint32_t current_bit = 1 << k + radixShift;
                asm("{\n"
                    "    .reg .pred p;\n"
                    "    and.b32 %0, %1, %2;"
                    "    setp.ne.u32 p, %0, 0;\n"
                    "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
                    "    @!p not.b32 %0, %0;\n"
                    "}\n" : "=r"(mask) : "r"(keys[i]), "r"(current_bit));
                warpFlags = (k == 0) ? mask : warpFlags & mask;
            }
            const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
            */
            unsigned warpFlags = 0xffffffff;
            #pragma unroll
            for (int k = 0; k < RADIX_LOG; ++k)
            {
                const bool t2 = keys[i] >> k + radixShift & 1;
                warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
            }
            const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

            //A roughly analogous version to what is used in D3D12 implementation.
            //Unfortunately, what we use here does not play well in HLSL
            /*
            offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits;
            __syncwarp(0xffffffff);
            if (bits == 0)
                s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
            __syncwarp(0xffffffff);
            */
            uint32_t preIncrementVal;
            if(bits == 0)
                preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(warpFlags));

            offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
        }
        __syncthreads();

        //exclusive prefix sum up the warp histograms
        if (threadIdx.x < RADIX)
        {
            uint32_t reduction = s_warpHistograms[threadIdx.x];
            for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX)
            {
                reduction += s_warpHistograms[i];
                s_warpHistograms[i] = reduction - s_warpHistograms[i];
            }

            atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
                    FLAG_REDUCTION | reduction << 2);

            //begin the exclusive prefix sum across the reductions
            s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
        }
        __syncthreads();

        if (threadIdx.x < (RADIX >> LANE_LOG))
            s_localHistogram[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_localHistogram[threadIdx.x << LANE_LOG]);
        __syncthreads();

        if (threadIdx.x < RADIX && getLaneId())
            s_localHistogram[threadIdx.x] += __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
        __syncthreads();

        //update offsets
        if (WARP_INDEX)
        {
            #pragma unroll 
            for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            {
                const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
                offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
            }
        }
        else
        {
            #pragma unroll
            for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
                offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
        }
        __syncthreads();

        //scatter keys into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            s_warpHistograms[offsets[i]] = keys[i];
  
        //split the warps into single thread cooperative groups and lookback
        if (threadIdx.x < RADIX)
        {
            uint32_t reduction = 0;
            for (uint32_t k = partitionIndex; k >= 0; )
            {
                const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX];

                if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    reduction += flagPayload >> 2;
                    atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX], 1 | (reduction << 2));
                    s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
                    break;
                }

                if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
                {
                    reduction += flagPayload >> 2;
                    k--;
                }
            }
        }
        __syncthreads();

        //scatter runs of keys into device memory
        #pragma unroll BIN_KEYS_PER_THREAD
        for (uint32_t i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
            alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
    }
    
    //Process the final partition slightly differently
    if(partitionIndex == gridDim.x - 1)
    {
        //immediately begin lookback
        if (threadIdx.x < RADIX)
        {
            if (partitionIndex)
            {
                uint32_t reduction = 0;
                for (uint32_t k = partitionIndex; k >= 0; )
                {
                    const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX];

                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        reduction += flagPayload >> 2;
                        s_localHistogram[threadIdx.x] = reduction;
                        break;
                    }

                    if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
                    {
                        reduction += flagPayload >> 2;
                        k--;
                    }
                }
            }
            else
            {
                s_localHistogram[threadIdx.x] = passHistogram[threadIdx.x] >> 2;
            }
        }
        __syncthreads();
        
        const uint32_t partEnd = BIN_PART_START + BIN_PART_SIZE;
        for (uint32_t i = threadIdx.x + BIN_PART_START; i < partEnd; i += blockDim.x)
        {
            uint32_t key;
            uint32_t offset;
            unsigned warpFlags = 0xffffffff;

            if(i < size)
                key = sort[i];

            #pragma unroll
            for (uint32_t k = 0; k < RADIX_LOG; ++k)
            {
                const bool t = key >> k + radixShift & 1;
                warpFlags &= (t ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t);
            }
            const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

            #pragma unroll
            for (uint32_t k = 0; k < BIN_WARPS; ++k)
            {
                uint32_t preIncrementVal;
                if (WARP_INDEX == k && bits == 0 && i < size)
                    preIncrementVal = atomicAdd(&s_localHistogram[key >> radixShift & RADIX_MASK], __popc(warpFlags));

                if (WARP_INDEX == k)
                    offset = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
                __syncthreads();
            }

            if(i < size)
                alt[offset] = key;
        }
    }
}