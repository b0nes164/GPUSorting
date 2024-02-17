/******************************************************************************
 * OneSweep Implementation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Author:  Thomas Smith 2/13/2024
 *
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//General macros 
#define PART_SIZE           3840U   //size of a partition tile

#define G_HIST_DIM          128U    //The number of threads in a global hist threadblock 
#define PASS_DIM            256U    //The number of threads int digit binning pass

#define RADIX               256U    //Number of digit bins
#define RADIX_MASK          255U    //Mask of digit bins
#define RADIX_LOG           8U      //log2(RADIX)
#define RADIX_PASSES        4U      //(Key width) / RADIX_LOG

#define HALF_RADIX          128U    //For smaller waves where bit packing is necessary
#define HALF_MASK           127U    // '' 

#define SEC_RADIX_START     256     //Offset for retrieving value from global histogram buffer
#define THIRD_RADIX_START   512     //Offset for retrieving value from global histogram buffer
#define FOURTH_RADIX_START  768     //Offset for retrieving value from global histogram buffer

//For the DigitBinningPass kernel
#define KEYS_PER_THREAD     15U     //The number of keys per thread in a DigitBinningPass threadblock
#define MAX_PASS_SMEM       4096U   //shared memory for DigitBinningPass kernel

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3       //Mask used to retrieve flag values

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_radixShift;
    uint e_threadBlocks;
    uint padding;
};

RWStructuredBuffer<uint> b_sort                         : register(u0); //buffer to sort
RWStructuredBuffer<uint> b_alt                          : register(u1); //double buffer
RWStructuredBuffer<uint> b_globalHist                   : register(u2); //buffer holding device level offsets for each binning pass
globallycoherent RWStructuredBuffer<uint> b_passHist    : register(u3); //buffer used to store reduced sums of partition tiles
globallycoherent RWStructuredBuffer<uint> b_index       : register(u4); //buffer used to atomically assign partition tile indexes

groupshared uint4 g_gHist[RADIX * 2];   //Shared memory for upsweep
groupshared uint g_pass[MAX_PASS_SMEM]; //Shared memory for the downsweep

inline uint getWaveIndex(uint gtid)
{
    return gtid / WaveGetLaneCount();
}

inline uint getWaveCountPass()
{
    return PASS_DIM / WaveGetLaneCount();
}

inline uint ExtractDigit(uint key)
{
    return key >> e_radixShift & RADIX_MASK;
}

inline uint ExtractDigit(uint key, uint shift)
{
    return key >> shift & RADIX_MASK;
}

inline uint ExtractPackedIndex(uint key)
{
    return key >> (e_radixShift + 1) & HALF_MASK;
}

inline uint ExtractPackedShift(uint key)
{
    return (key >> e_radixShift & 1) ? 16 : 0;
}

inline uint ExtractPackedValue(uint packed, uint key)
{
    return packed >> ExtractPackedShift(key) & 0xffff;
}

inline uint SubPartSizeWGE16()
{
    return KEYS_PER_THREAD * WaveGetLaneCount();
}

inline uint SharedOffsetWGE16(uint gtid)
{
    return WaveGetLaneIndex() + getWaveIndex(gtid) * SubPartSizeWGE16();
}

inline uint DeviceOffsetWGE16(uint gtid, uint gid)
{
    return SharedOffsetWGE16(gtid) + gid * PART_SIZE;
}

inline uint SubPartSizeWLT16(uint _serialIterations)
{
    return KEYS_PER_THREAD * WaveGetLaneCount() * _serialIterations;
}

inline uint SharedOffsetWLT16(uint gtid, uint _serialIterations)
{
    return WaveGetLaneIndex() +
        (getWaveIndex(gtid) / _serialIterations * SubPartSizeWLT16(_serialIterations)) +
        (getWaveIndex(gtid) % _serialIterations * WaveGetLaneCount());
}

inline uint DeviceOffsetWLT16(uint gtid, uint gid, uint _serialIterations)
{
    return SharedOffsetWLT16(gtid, _serialIterations) + gid * PART_SIZE;
}

inline uint PassHistOffset(uint digit)
{
    return (digit + (e_radixShift >> 3 << RADIX_LOG)) * e_threadBlocks;
}

inline uint WaveHistsSizeWGE16()
{
    return PASS_DIM / WaveGetLaneCount() * RADIX;
}

inline uint WaveHistsSizeWLT16()
{
    return MAX_PASS_SMEM;
}

[numthreads(256, 1, 1)]
void InitOneSweep(uint3 id : SV_DispatchThreadID)
{
    const uint increment = 256 * 256;
    const uint clearEnd = e_threadBlocks * RADIX * RADIX_PASSES;
    for (uint i = id.x; i < clearEnd; i += increment)
        b_passHist[i] = 0;

    if(id.x < RADIX * RADIX_PASSES)
        b_globalHist[id.x] = 0;
    
    if (id.x < RADIX_PASSES)
        b_index[id.x] = 0;
}

[numthreads(G_HIST_DIM, 1, 1)]
void GlobalHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    //clear shared memory
    const uint histsEnd = RADIX * 2;
    for (uint i = gtid.x; i < histsEnd; i += G_HIST_DIM)
        g_gHist[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    
    //histogram, 64 threads to a histogram
    const uint histOffset = gtid.x / 64 * RADIX;
    const uint partitionEnd = gid.x == e_threadBlocks - 1 ?
        e_numKeys : (gid.x + 1) * PART_SIZE;
    for (uint i = gtid.x + gid.x * PART_SIZE; i < partitionEnd; i += G_HIST_DIM)
    {
        const uint t = b_sort[i];
        InterlockedAdd(g_gHist[ExtractDigit(t, 0)].x, 1);
        InterlockedAdd(g_gHist[ExtractDigit(t, 8)].y, 1);
        InterlockedAdd(g_gHist[ExtractDigit(t, 16)].z, 1);
        InterlockedAdd(g_gHist[ExtractDigit(t, 24)].w, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    
    //reduce
    for (uint i = gtid.x; i < RADIX; i += G_HIST_DIM)
        g_gHist[i] += g_gHist[i + RADIX];
    
    //waves 16 or greater can perform a more elegant scan because 16 * 16 = 256
    if (WaveGetLaneCount() >= 16)
    {
        for (uint i = gtid.x; i < RADIX; i += G_HIST_DIM)
            g_gHist[i] += WavePrefixSum(g_gHist[i]);
        GroupMemoryBarrierWithGroupSync();
        
        if (gtid.x < (RADIX / WaveGetLaneCount()))
        {
            g_gHist[(gtid.x + 1) * WaveGetLaneCount() - 1] +=
                WavePrefixSum(g_gHist[(gtid.x + 1) * WaveGetLaneCount() - 1]);
        }
        GroupMemoryBarrierWithGroupSync();
        
        //atomically add to global histogram
        const uint laneMask = WaveGetLaneCount() - 1;
        const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
        for (uint i = gtid.x; i < RADIX; i += G_HIST_DIM)
        {
            const uint index = circularLaneShift + (i & ~laneMask);
            
            InterlockedAdd(b_globalHist[index],
                (WaveGetLaneIndex() != laneMask ? g_gHist[i].x : 0) +
                (i >= WaveGetLaneCount() ? WaveReadLaneAt(g_gHist[i - 1].x, 0) : 0));
            
            InterlockedAdd(b_globalHist[index + SEC_RADIX_START],
                (WaveGetLaneIndex() != laneMask ? g_gHist[i].y : 0) +
                (i >= WaveGetLaneCount() ? WaveReadLaneAt(g_gHist[i - 1].y, 0) : 0));
            
            InterlockedAdd(b_globalHist[index + THIRD_RADIX_START],
                (WaveGetLaneIndex() != laneMask ? g_gHist[i].z : 0) +
                (i >= WaveGetLaneCount() ? WaveReadLaneAt(g_gHist[i - 1].z, 0) : 0));
            
            InterlockedAdd(b_globalHist[index + FOURTH_RADIX_START],
                (WaveGetLaneIndex() != laneMask ? g_gHist[i].w : 0) +
                (i >= WaveGetLaneCount() ? WaveReadLaneAt(g_gHist[i - 1].w, 0) : 0));
        }
    }
    
    //exclusive Brent-Kung with fused upsweep downsweep
    if (WaveGetLaneCount() < 16)
    {
        //TODO
    }
}

[numthreads(PASS_DIM, 1, 1)]
void DigitBinningPass(uint3 gtid : SV_GroupThreadID)
{
    //clear shared memory
    const uint histsEnd = WaveGetLaneCount() < 16 ? WaveHistsSizeWLT16() : WaveHistsSizeWGE16();
    for (uint i = gtid.x; i < histsEnd; i += PASS_DIM)
        g_pass[i] = 0;

    //atomically assign partition tile index
    if(gtid.x == 0)
        InterlockedAdd(b_index[e_radixShift >> 3], 1, g_pass[PART_SIZE - 1]);
    GroupMemoryBarrierWithGroupSync();
    const uint partitionIndex = g_pass[PART_SIZE - 1];
    
    if (true)
    {
        uint keys[KEYS_PER_THREAD];
#if defined(ENABLE_16_BIT)
        uint16_t offsets[KEYS_PER_THREAD];
#else
        uint offsets[KEYS_PER_THREAD];
#endif
        
        if (WaveGetLaneCount() >= 16)
        {
            //Load keys into registers
            [unroll]
            for (uint i = 0, t = DeviceOffsetWGE16(gtid.x, partitionIndex);
                 i < KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
            {
                keys[i] = b_sort[t];
            }
            
            
            //Warp Level Multisplit
            const uint waveParts = (WaveGetLaneCount() + 31) / 32;
            [unroll]
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            {
                uint4 waveFlags = (WaveGetLaneCount() & 31) ?
                    (1U << WaveGetLaneCount()) - 1 : 0xffffffff;

                [unroll]
                for (uint k = 0; k < RADIX_LOG; ++k)
                {
                    const bool t = keys[i] >> (k + e_radixShift) & 1;
                    const uint4 ballot = WaveActiveBallot(t);
                    for (int wavePart = 0; wavePart < waveParts; ++wavePart)
                        waveFlags[wavePart] &= (t ? 0 : 0xffffffff) ^ ballot[wavePart];
                }
                    
                uint bits = 0;
                for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                {
                    if (WaveGetLaneIndex() >= wavePart * 32)
                    {
                        const uint ltMask = WaveGetLaneIndex() >= (wavePart + 1) * 32 ?
                            0xffffffff : (1U << (WaveGetLaneIndex() & 31)) - 1;
                        bits += countbits(waveFlags[wavePart] & ltMask);
                    }
                }
                    
                const uint index = ExtractDigit(keys[i]) + (getWaveIndex(gtid.x) * RADIX);
                offsets[i] = g_pass[index] + bits;
                    
                GroupMemoryBarrierWithGroupSync();
                if (bits == 0)
                {
                    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
                        g_pass[index] += countbits(waveFlags[wavePart]);
                }
                GroupMemoryBarrierWithGroupSync();
            }
            
            //inclusive/exclusive prefix sum up the histograms
            //followed by exclusive prefix sum across the reductions
            uint reduction = g_pass[gtid.x];
            for (uint i = gtid.x + RADIX; i < WaveHistsSizeWGE16(); i += RADIX)
            {
                reduction += g_pass[i];
                g_pass[i] = reduction - g_pass[i];
            }
            
            InterlockedAdd(b_passHist[PassHistOffset(gtid.x) + partitionIndex],
                (partitionIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | reduction << 2);
            reduction += WavePrefixSum(reduction);
            GroupMemoryBarrierWithGroupSync();

            const uint laneMask = WaveGetLaneCount() - 1;
            g_pass[((WaveGetLaneIndex() + 1) & laneMask) + (gtid.x & ~laneMask)] = reduction;
            GroupMemoryBarrierWithGroupSync();
                
            if (gtid.x < RADIX / WaveGetLaneCount())
            {
                g_pass[gtid.x * WaveGetLaneCount()] =
                    WavePrefixSum(g_pass[gtid.x * WaveGetLaneCount()]);
            }
            GroupMemoryBarrierWithGroupSync();
                
            if (WaveGetLaneIndex())
                g_pass[gtid.x] += WaveReadLaneAt(g_pass[gtid.x - 1], 1);
            GroupMemoryBarrierWithGroupSync();
            
            //Update offsets
            if (gtid.x >= WaveGetLaneCount())
            {
                const uint t = getWaveIndex(gtid.x) * RADIX;
                [unroll]
                for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                {
                    const uint t2 = ExtractDigit(keys[i]);
                    offsets[i] += g_pass[t2 + t] + g_pass[t2];
                }
            }
            else
            {
                [unroll]
                for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                    offsets[i] += g_pass[ExtractDigit(keys[i])];
            }
            
            //lookback
            //Extremely slow
            /*
            if(partitionIndex)
            {
                uint reduction = 0;
                const uint passHistOffset = PassHistOffset(gtid.x);
                for (uint k = partitionIndex; k > 0;)
                {
                    const uint flagPayload = b_passHist[passHistOffset + k - 1];
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        reduction += flagPayload >> 2;
                        InterlockedAdd(b_passHist[passHistOffset + partitionIndex], 1 | (reduction << 2));
                        g_pass[gtid.x + PART_SIZE] = b_globalHist[gtid.x + (e_radixShift << 5)] +
                            reduction - g_pass[gtid.x];
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
                g_pass[gtid.x + PART_SIZE] = b_globalHist[gtid.x + (e_radixShift << 5)] -
                    g_pass[gtid.x];
            }
            */
            
            if(partitionIndex)
            {
                for (uint i = getWaveIndex(gtid.x); i < RADIX; i += getWaveCountPass())
                {
                    uint reduction = 0;
                    const uint passHistOffset = PassHistOffset(i);
                    for (int k = (int)partitionIndex - WaveGetLaneIndex(); k > 0;)
                    {
                        const uint flagPayload = b_passHist[passHistOffset + k - 1];
                        if (WaveActiveAllTrue((flagPayload & FLAG_MASK) > FLAG_NOT_READY))
                        {
                            const uint inclusiveIndex = firstbitlow((uint)WaveActiveBallot((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE));
                            if (inclusiveIndex != 0xffffffff)
                            {
                                reduction += WaveActiveSum(WaveGetLaneIndex() <= inclusiveIndex ? (flagPayload >> 2) : 0);
                                
                                if (WaveGetLaneIndex() == inclusiveIndex)
                                {
                                    InterlockedAdd(b_passHist[passHistOffset + partitionIndex], 1 | (reduction << 2));
                                    g_pass[i + PART_SIZE] = b_globalHist[i + (e_radixShift << 5)] +
                                        reduction - g_pass[i];
                                }
                                break;
                            }
                            else
                            {
                                reduction += WaveActiveSum(flagPayload >> 2);
                                k -= WaveGetLaneCount();
                            }
                        }
                    }
                }
            }
            else
            {
                g_pass[gtid.x + PART_SIZE] = b_globalHist[gtid.x + (e_radixShift << 5)] -
                    g_pass[gtid.x];
            }
            GroupMemoryBarrierWithGroupSync();

            //scatter keys into shared memory
            for (uint i = 0; i < KEYS_PER_THREAD; ++i)
                g_pass[offsets[i]] = keys[i];
            GroupMemoryBarrierWithGroupSync();
            
            //scatter keys into device
            for (uint i = gtid.x; i < PART_SIZE; i += PASS_DIM)
            {
                b_alt[g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i] = g_pass[i];
            }
        }
        
        if (WaveGetLaneCount() < 16)
        {
            
        }
    }
}