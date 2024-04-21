/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/19/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#include "EmulatedDeadlocking.cuh"

#define RADIX               256
#define RADIX_MASK          255
#define RADIX_LOG           8

#define SEC_RADIX_START     256
#define THIRD_RADIX_START   512
#define FOURTH_RADIX_START  768

#define G_HIST_PART_SIZE	65536
#define G_HIST_VEC_SIZE		16384

#define BIN_PART_SIZE       7680
#define BIN_HISTS_SIZE      4096
#define BIN_SUB_PART_SIZE   480
#define BIN_WARPS           16
#define BIN_KEYS_PER_THREAD 15
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)

#define FLAG_NOT_READY      0
#define FLAG_REDUCTION      1
#define FLAG_INCLUSIVE      2
#define FLAG_MASK           3

//To emulate deadlocking
#define MASK                8191
#define MAX_SPIN_COUNT      1

__global__ void EmulatedDeadlocking::GlobalHistogram(
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

__global__ void EmulatedDeadlocking::Scan(
    uint32_t* globalHistogram,
    uint32_t* firstPassHistogram,
    uint32_t* secPassHistogram,
    uint32_t* thirdPassHistogram,
    uint32_t* fourthPassHistogram)
{
    __shared__ uint32_t s_scan[RADIX];

    s_scan[threadIdx.x] = InclusiveWarpScanCircularShift(globalHistogram[threadIdx.x + blockIdx.x * RADIX]);
    __syncthreads();

    if (threadIdx.x < (RADIX >> LANE_LOG))
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

__device__ __forceinline__ void LookbackWithFallback(
    uint32_t* sort,
    volatile uint32_t* passHistogram,
    uint32_t& lock,
    uint32_t& deadlockEncountered,
    uint32_t* histogram,
    const uint32_t& partIndex,
    const uint32_t& radixShift,
    const uint32_t& exclusiveHistReduction)
{
    uint32_t spinCount = 0;
    uint32_t reduction = 0;
    bool lookbackComplete = threadIdx.x < RADIX ? false : true;
    bool warpLookbackComplete = false;
    uint32_t lookbackIndex = (threadIdx.x & RADIX_MASK) + partIndex * RADIX;

    while (lock < BIN_WARPS)
    {
        //Try to read the preceeding tiles
        uint32_t flagPayload;
        if (!warpLookbackComplete)
        {
            if (!lookbackComplete)
            {
                while (spinCount < MAX_SPIN_COUNT)
                {
                    flagPayload = passHistogram[lookbackIndex];
                    if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY)
                        break;
                    else
                        spinCount++;
                }
            }

            //Did we encounter any deadlocks?
            const bool deadlock = __any_sync(0xffffffff, spinCount == MAX_SPIN_COUNT);
            if (!getLaneId() && deadlock)
                atomicOr((uint32_t*)&deadlockEncountered, 1);
        }
        __syncthreads();

        //Yes: fallback
        if (deadlockEncountered)
        {
            if(threadIdx.x < RADIX)
                histogram[threadIdx.x] = 0;
            __syncthreads();
            if (!threadIdx.x)
                deadlockEncountered = 0;

            const uint32_t fallbackEnd = BIN_PART_SIZE * (lookbackIndex >> RADIX_LOG);
            for (uint32_t i = threadIdx.x + BIN_PART_SIZE * ((lookbackIndex >> RADIX_LOG) - 1); i < fallbackEnd; i += blockDim.x)
                atomicAdd((uint32_t*)&histogram[sort[i] >> radixShift & RADIX_MASK], 1);
            __syncthreads();

            uint32_t reduceOut;
            if (threadIdx.x < RADIX)
            {
                reduceOut = atomicCAS((uint32_t*)&passHistogram[threadIdx.x + (lookbackIndex >> RADIX_LOG) * RADIX], 0,
                    FLAG_REDUCTION | histogram[threadIdx.x] << 2);
            }

            if (!lookbackComplete)
            {
                if ((reduceOut & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    reduction += reduceOut >> 2;
                    atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partIndex + 1) * RADIX], 1 | (reduction << 2));
                    lookbackComplete = true;
                }
                else
                {
                    reduction += histogram[threadIdx.x];
                }
            }
            spinCount = 0;
        }
        else //No: proceed as normal
        {
            if (!lookbackComplete)
            {
                reduction += flagPayload >> 2;
                if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                {
                    atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partIndex + 1) * RADIX], 1 | (reduction << 2));
                    lookbackComplete = true;
                }
                else
                {
                    spinCount = 0;
                }
            }
        }
        lookbackIndex -= RADIX; //The threadblock lookbacks in lockstep.
        
        //Have all digits completed their lookbacks?
        if (!warpLookbackComplete)
        {
            warpLookbackComplete = __all_sync(0xffffffff, lookbackComplete);
            if (!getLaneId() && warpLookbackComplete)
                atomicAdd((uint32_t*)&lock, 1);
        }
        __syncthreads();
    }

    //post results into shared memory
    if(threadIdx.x < RADIX)
        histogram[threadIdx.x] = reduction - exclusiveHistReduction;
}

__global__ void EmulatedDeadlocking::EmulatedDeadlockingSpinning(
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

    //load keys
    uint32_t keys[BIN_KEYS_PER_THREAD];
    if (partitionIndex < gridDim.x - 1)
    {
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = sort[t];
    }

    if (partitionIndex == gridDim.x - 1)
    {
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = t < size ? sort[t] : 0xffffffff;
    }

    //WLMS
    uint16_t offsets[BIN_KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    {
        unsigned warpFlags = 0xffffffff;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k)
        {
            const bool t2 = keys[i] >> k + radixShift & 1;
            warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }
        const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

        uint32_t preIncrementVal;
        if (bits == 0)
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

        if (!(partitionIndex & MASK) && partitionIndex != gridDim.x - 1)
        {
            while (passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX] == 0)
            {
                __threadfence();
            }
        }
        else
        {
            atomicCAS((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX], 0,
                FLAG_REDUCTION | reduction << 2);
        }

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

    //take advantage of the barrier to read keys and set locks for the lookback
    uint32_t exclusiveHistReduction;
    if(threadIdx.x < RADIX)
        exclusiveHistReduction = s_localHistogram[threadIdx.x];

    if (!threadIdx.x)
    {
        s_warpHistograms[0] = 0;
        s_warpHistograms[1] = 0;
    }
    __syncthreads();

    LookbackWithFallback(
        sort,
        passHistogram,
        s_warpHistograms[0],
        s_warpHistograms[1],
        s_localHistogram,
        partitionIndex,
        radixShift,
        exclusiveHistReduction);
    __syncthreads();

    //scatter keys into shared memory
    #pragma unroll
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        s_warpHistograms[offsets[i]] = keys[i];
    __syncthreads();

    //scatter runs of keys into device memory
    if (partitionIndex < gridDim.x - 1)
    {
        #pragma unroll BIN_KEYS_PER_THREAD
        for (uint32_t i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
            alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
    }

    if (partitionIndex == gridDim.x - 1)
    {
        const uint32_t finalPartSize = size - BIN_PART_START;
        for (uint32_t i = threadIdx.x; i < finalPartSize; i += blockDim.x)
            alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
    }
}