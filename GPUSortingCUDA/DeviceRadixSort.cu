/******************************************************************************
 * GPUSorting
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#include "DeviceRadixSort.cuh"

#define RADIX               256     //Number of digit bins
#define RADIX_MASK          255     //Mask of digit bins, to extract digits
#define RADIX_LOG           8       //log2(RADIX)

#define SEC_RADIX_START     256     //Offset for retrieving value from global histogram buffer
#define THIRD_RADIX_START   512     //Offset for retrieving value from global histogram buffer
#define FOURTH_RADIX_START  768     //Offset for retrieving value from global histogram buffer

//For the upfront global histogram kernel
#define PART_SIZE			7680
#define VEC_PART_SIZE		1920

//For the digit binning
#define BIN_PART_SIZE       7680                                    //Partition tile size in k_DigitBinning
#define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
#define BIN_SUB_PART_SIZE   480                                     //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_WARPS           16                                      //Warps per threadblock in k_DigitBinning
#define BIN_KEYS_PER_THREAD 15                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (blockIdx.x * BIN_PART_SIZE)			//Starting offset of a partition tile

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0                                       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1                                       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2                                       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3                                       //Mask used to retrieve flag values

__global__ void DeviceRadixSort::Upsweep(
	uint32_t* sort,
	uint32_t* globalHist,
	uint32_t* passHist,
	uint32_t size,
	uint32_t radixShift)
{
	__shared__ uint32_t s_globalHist[RADIX * 2];

	//clear shared memory
	for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
		s_globalHist[i] = 0;
	__syncthreads();
	
	//histogram
	{
		//64 threads : 1 histogram in shared memory
		uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];

		if (blockIdx.x < gridDim.x - 1)
		{
			const uint32_t partEnd = (blockIdx.x + 1) * VEC_PART_SIZE;
			for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x)
			{
				const uint4 t = reinterpret_cast<uint4*>(sort)[i];
				atomicAdd(&s_wavesHist[t.x >> radixShift & RADIX_MASK], 1);
				atomicAdd(&s_wavesHist[t.y >> radixShift & RADIX_MASK], 1);
				atomicAdd(&s_wavesHist[t.z >> radixShift & RADIX_MASK], 1);
				atomicAdd(&s_wavesHist[t.w >> radixShift & RADIX_MASK], 1);
			}
		}

		if (blockIdx.x == gridDim.x - 1)
		{
			for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < size; i += blockDim.x)
			{
				const uint32_t t = sort[i];
				atomicAdd(&s_wavesHist[t >> radixShift & RADIX_MASK], 1);
			}
		}
	}
	__syncthreads();

	//reduce to the first hist, pass out, begin prefix sum
	for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
	{
		s_globalHist[i] += s_globalHist[i + RADIX];
		passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
		s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
	}	
	__syncthreads();

	if (threadIdx.x < (RADIX >> LANE_LOG))
		s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
	__syncthreads();
	
	//Atomically add to device memory
	for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
		atomicAdd(&globalHist[i + (radixShift << 5)], s_globalHist[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0));
}

__global__ void DeviceRadixSort::Scan(
	uint32_t* passHist,
	uint32_t threadBlocks)
{
	__shared__ uint32_t s_scan[128];

	uint32_t reduction = 0;
	const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;
	const uint32_t partitionsEnd = threadBlocks / blockDim.x * blockDim.x;
	const uint32_t digitOffset = blockIdx.x * threadBlocks;

	uint32_t i = threadIdx.x;
	for (; i < partitionsEnd; i += blockDim.x)
	{
		s_scan[threadIdx.x] = passHist[i + digitOffset];
		s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
		__syncthreads();

		if (threadIdx.x < (blockDim.x >> LANE_LOG))
		{
			s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] = 
				ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
		}
		__syncthreads();

		passHist[circularLaneShift + (i & ~LANE_MASK) + digitOffset] =
			(getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
			(threadIdx.x >= LANE_COUNT ? __shfl_sync(0xffffffff, s_scan[threadIdx.x - 1], 0) : 0) +
			reduction;

		reduction += s_scan[blockDim.x - 1];
		__syncthreads();
	}

	if(i < threadBlocks)
		s_scan[threadIdx.x] = passHist[i + digitOffset];
	s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
	__syncthreads();

	if (threadIdx.x < (blockDim.x >> LANE_LOG))
	{
		s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] =
			ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
	}
	__syncthreads();

	const uint32_t index = circularLaneShift + (i & ~LANE_MASK);
	if (index < threadBlocks)
	{
		passHist[index + digitOffset] =
			(getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
			(threadIdx.x >= LANE_COUNT ?
			s_scan[(threadIdx.x & ~LANE_MASK) - 1] : 0) +
			reduction;
	}
}

__global__ void DeviceRadixSort::DownsweepKeysOnly(
	uint32_t* sort, 
	uint32_t* alt, 
	uint32_t* globalHist,
	uint32_t* passHist,
	uint32_t size, 
	uint32_t radixShift)
{
	__shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
	__shared__ uint32_t s_localHistogram[RADIX];
	volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

	//clear shared memory
	for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
		s_warpHistograms[i] = 0;

	//load keys
	uint32_t keys[BIN_KEYS_PER_THREAD];
	if (blockIdx.x < gridDim.x - 1)
	{
		#pragma unroll
		for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
			keys[i] = sort[t];
	}

	//To handle input sizes not perfect multiples of the partition tile size,
	//load "dummy" keys, which are keys with the highest possible digit.
	//Because of the stability of the sort, these keys are guaranteed to be 
	//last when scattered. This allows for effortless divergence free sorting
	//of the final partition.
	if (blockIdx.x == gridDim.x - 1)
	{
		#pragma unroll
		for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
			keys[i] = t < size ? sort[t] : 0xffffffff;
	}
	__syncthreads();

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

		//begin the exclusive prefix sum across the reductions
		s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
	}
	__syncthreads();

	if (threadIdx.x < (RADIX >> LANE_LOG))
		s_warpHistograms[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
	__syncthreads();

	if (threadIdx.x < RADIX && getLaneId())
		s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
	__syncthreads();

	//update offsets
	if (WARP_INDEX)
	{
		#pragma unroll 
		for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		{
			const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
			offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
		}
	}
	else
	{
		#pragma unroll
		for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
			offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
	}

	//load in threadblock reductions
	if (threadIdx.x < RADIX)
	{
		s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
			passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
	}
	__syncthreads();

	//scatter keys into shared memory
	#pragma unroll
	for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		s_warpHistograms[offsets[i]] = keys[i];
	__syncthreads();

	//scatter runs of keys into device memory
	if (blockIdx.x < gridDim.x - 1)
	{
		#pragma unroll BIN_KEYS_PER_THREAD
		for (uint32_t i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
			alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
	}

	if (blockIdx.x == gridDim.x - 1)
	{
		const uint32_t finalPartSize = size - BIN_PART_START;
		for (uint32_t i = threadIdx.x; i < finalPartSize; i += blockDim.x)
			alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
	}
}

__global__ void DeviceRadixSort::DownsweepPairs(
	uint32_t* sort,
	uint32_t* sortPayload,
	uint32_t* alt, 
	uint32_t* altPayload,
	uint32_t* globalHist,
	uint32_t* passHist,
	uint32_t size, 
	uint32_t radixShift)
{
	__shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
	__shared__ uint32_t s_localHistogram[RADIX];
	volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

	//clear shared memory
	for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
		s_warpHistograms[i] = 0;

	//load keys
	uint32_t keys[BIN_KEYS_PER_THREAD];
	if (blockIdx.x < gridDim.x - 1)
	{
		#pragma unroll
		for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
			keys[i] = sort[t];
	}

	//To handle input sizes not perfect multiples of the partition tile size,
	//load "dummy" keys, which are keys with the highest possible digit.
	//Because of the stability of the sort, these keys are guaranteed to be 
	//last when scattered. This allows for effortless divergence free sorting
	//of the final partition.
	if (blockIdx.x == gridDim.x - 1)
	{
		#pragma unroll
		for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
			keys[i] = t < size ? sort[t] : 0xffffffff;
	}
	__syncthreads();

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

		//begin the exclusive prefix sum across the reductions
		s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
	}
	__syncthreads();

	if (threadIdx.x < (RADIX >> LANE_LOG))
		s_warpHistograms[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
	__syncthreads();

	if (threadIdx.x < RADIX && getLaneId())
		s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
	__syncthreads();

	//update offsets
	if (WARP_INDEX)
	{
		#pragma unroll 
		for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		{
			const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
			offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
		}
	}
	else
	{
		#pragma unroll
		for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
			offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
	}

	//load in threadblock reductions
	if (threadIdx.x < RADIX)
	{
		s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
			passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
	}
	__syncthreads();

	//scatter keys into shared memory
	#pragma unroll
	for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		s_warpHistograms[offsets[i]] = keys[i];
	__syncthreads();

	//scatter runs of keys into device memory
	if (blockIdx.x < gridDim.x - 1)
    {
        //store the scatter index in key register to reuse for payloads
        //scatter in the same warp striped format as loaded in, to reuse 
        //the offsets we already calculated.
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            keys[i] = s_localHistogram[s_warpHistograms[t] >> radixShift & RADIX_MASK] + t;
            alt[keys[i]] = s_warpHistograms[t];
        }
        __syncthreads();

        //Reuse the offsets to scatter payloads in order directly from device memory
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            s_warpHistograms[offsets[i]] = sortPayload[t];
        }
        __syncthreads();

        //Now scatter the payloads using the values stored in the keys reg
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            altPayload[keys[i]] = s_warpHistograms[t];
        }
    }

	if (blockIdx.x == gridDim.x - 1)
    {
        //In the final partition, each thread must determine how many keys it will process
        uint32_t finalKeys;
        int32_t subPartSize = (int32_t)(size - BIN_PART_START) - (int32_t)BIN_SUB_PART_START;
        if (subPartSize > 0)
        {
            if ((uint32_t)subPartSize >= BIN_SUB_PART_SIZE)
            {
                finalKeys = BIN_KEYS_PER_THREAD;
            }
            else
            {
                finalKeys = subPartSize / LANE_COUNT;
                subPartSize -= (int32_t)(finalKeys * LANE_COUNT); //cannot go below zero
                if (getLaneId() < (uint32_t)subPartSize)
                    finalKeys++;
            }
        }
        else
        {
            finalKeys = 0;
        }

        //store the scatter index in key register to reuse for payloads
        //scatter in the same warp striped format as loaded in, to reuse 
        //the offsets we already calculated.
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            if (i < finalKeys)
            {
                keys[i] = s_localHistogram[s_warpHistograms[t] >> radixShift & RADIX_MASK] + t;
                alt[keys[i]] = s_warpHistograms[t];
            }
        }
        __syncthreads();

        //Reuse the offsets to scatter payloads in order directly from device memory
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            if (i < finalKeys)
                s_warpHistograms[offsets[i]] = sortPayload[t];
        }
        __syncthreads();

        //Now scatter the payloads using the values stored in the keys reg
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            if (i < finalKeys)
                altPayload[keys[i]] = s_warpHistograms[t];
        }
    }
}