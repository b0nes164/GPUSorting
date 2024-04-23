/******************************************************************************
 * GPUSorting
 * Emulated Deadlocking Testing, used to emulate the behaviour of GPU's without
 * forward progress guaruntees.
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/22/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
#include "SweepCommon.hlsl"

#define MASK    8191

[numthreads(4, 1, 1)]
void ClearIndex(uint3 gtid : SV_GroupThreadID)
{
    b_index[gtid.x] = 0;
}

[numthreads(D_DIM, 1, 1)]
void EmulatedDeadlockingPassOne(uint3 gtid : SV_GroupThreadID)
{
    uint partitionIndex;
    KeyStruct keys;
    OffsetStruct offsets;
    
    //WGT 16 can potentially skip some barriers
    if (WaveGetLaneCount() > 16)
    {
        if (WaveHistsSizeWGE16() < PART_SIZE)
            ClearWaveHists(gtid.x);

        AssignPartitionTile(gtid.x, partitionIndex);
        if (WaveHistsSizeWGE16() >= PART_SIZE)
        {
            GroupMemoryBarrierWithGroupSync();
            ClearWaveHists(gtid.x);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    
    if ((partitionIndex & MASK))
    {
        if (WaveGetLaneCount() <= 16)
        {
            AssignPartitionTile(gtid.x, partitionIndex);
            GroupMemoryBarrierWithGroupSync();
            ClearWaveHists(gtid.x);
            GroupMemoryBarrierWithGroupSync();
        }
    
        if (partitionIndex < e_threadBlocks - 1)
        {
            if (WaveGetLaneCount() >= 16)
                keys = LoadKeysWGE16(gtid.x, partitionIndex);
        
            if (WaveGetLaneCount() < 16)
                keys = LoadKeysWLT16(gtid.x, partitionIndex, SerialIterations());
        }
        
        if (partitionIndex == e_threadBlocks - 1)
        {
            if (WaveGetLaneCount() >= 16)
                keys = LoadKeysPartialWGE16(gtid.x, partitionIndex);
        
            if (WaveGetLaneCount() < 16)
                keys = LoadKeysPartialWLT16(gtid.x, partitionIndex, SerialIterations());
        }
    
        uint exclusiveHistReduction;
        if (WaveGetLaneCount() >= 16)
        {
            offsets = RankKeysWGE16(gtid.x, keys);
            GroupMemoryBarrierWithGroupSync();
        
            uint histReduction;
            if (gtid.x < RADIX)
            {
                histReduction = WaveHistInclusiveScanCircularShiftWGE16(gtid.x);
                CASDeviceBroadcastReductionsWGE16(gtid.x, partitionIndex, histReduction);
                histReduction += WavePrefixSum(histReduction); //take advantage of barrier to begin scan
            }
            GroupMemoryBarrierWithGroupSync();

            WaveHistReductionExclusiveScanWGE16(gtid.x, histReduction);
            GroupMemoryBarrierWithGroupSync();
            
            UpdateOffsetsWGE16(gtid.x, offsets, keys);
            GroupMemoryBarrierWithGroupSync();
            
            if (gtid.x < RADIX)
                exclusiveHistReduction = g_d[gtid.x];   //take advantage of barrier to grab value
            InitializeLookbackFallback(gtid.x);         //Set the locks for lookback fallback in shared memory
            GroupMemoryBarrierWithGroupSync();
        }
    
        if (WaveGetLaneCount() < 16)
        {
            offsets = RankKeysWLT16(gtid.x, keys, SerialIterations());
            
            if (gtid.x < HALF_RADIX)
            {
                uint histReduction = WaveHistInclusiveScanCircularShiftWLT16(gtid.x);
                g_d[gtid.x] = histReduction + (histReduction << 16); //take advantage of barrier to begin scan
                CASDeviceBroadcastReductionsWLT16(gtid.x, partitionIndex, histReduction);
            }
            
            WaveHistReductionExclusiveScanWLT16(gtid.x);
            GroupMemoryBarrierWithGroupSync();
            
            UpdateOffsetsWLT16(gtid.x, SerialIterations(), offsets, keys);
            GroupMemoryBarrierWithGroupSync();
            
            if (gtid.x < RADIX)                 //take advantage of barrier to grab value
                exclusiveHistReduction = g_d[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
            InitializeLookbackFallback(gtid.x); //Set the locks for lookback fallback in shared memory
            GroupMemoryBarrierWithGroupSync();
        }
    
        LookbackWithFallback(gtid.x, partitionIndex, exclusiveHistReduction);
        GroupMemoryBarrierWithGroupSync();
        
        ScatterKeysShared(offsets, keys);
        GroupMemoryBarrierWithGroupSync();
        
        if (partitionIndex < e_threadBlocks - 1)
            ScatterDevice(gtid.x, partitionIndex, offsets);
        
        if (partitionIndex == e_threadBlocks - 1)
            ScatterDevicePartial(gtid.x, partitionIndex, offsets);
    }
}

[numthreads(D_DIM, 1, 1)]
void EmulatedDeadlockingPassTwo(uint3 gtid : SV_GroupThreadID)
{
    uint partitionIndex;
    KeyStruct keys;
    OffsetStruct offsets;
    
    //WGT 16 can potentially skip some barriers
    if (WaveGetLaneCount() > 16)
    {
        if (WaveHistsSizeWGE16() < PART_SIZE)
            ClearWaveHists(gtid.x);

        AssignPartitionTile(gtid.x, partitionIndex);
        if (WaveHistsSizeWGE16() >= PART_SIZE)
        {
            GroupMemoryBarrierWithGroupSync();
            ClearWaveHists(gtid.x);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    
    if (!(partitionIndex & MASK))
    {
        if (WaveGetLaneCount() <= 16)
        {
            AssignPartitionTile(gtid.x, partitionIndex);
            GroupMemoryBarrierWithGroupSync();
            ClearWaveHists(gtid.x);
            GroupMemoryBarrierWithGroupSync();
        }
    
        if (partitionIndex < e_threadBlocks - 1)
        {
            if (WaveGetLaneCount() >= 16)
                keys = LoadKeysWGE16(gtid.x, partitionIndex);
        
            if (WaveGetLaneCount() < 16)
                keys = LoadKeysWLT16(gtid.x, partitionIndex, SerialIterations());
        }
        
        if (partitionIndex == e_threadBlocks - 1)
        {
            if (WaveGetLaneCount() >= 16)
                keys = LoadKeysPartialWGE16(gtid.x, partitionIndex);
        
            if (WaveGetLaneCount() < 16)
                keys = LoadKeysPartialWLT16(gtid.x, partitionIndex, SerialIterations());
        }
    
        uint exclusiveHistReduction;
        if (WaveGetLaneCount() >= 16)
        {
            offsets = RankKeysWGE16(gtid.x, keys);
            GroupMemoryBarrierWithGroupSync();
        
            uint histReduction;
            if (gtid.x < RADIX)
            {
                histReduction = WaveHistInclusiveScanCircularShiftWGE16(gtid.x);
                CASDeviceBroadcastReductionsWGE16(gtid.x, partitionIndex, histReduction);
                histReduction += WavePrefixSum(histReduction); //take advantage of barrier to begin scan
            }
            GroupMemoryBarrierWithGroupSync();

            WaveHistReductionExclusiveScanWGE16(gtid.x, histReduction);
            GroupMemoryBarrierWithGroupSync();
            
            UpdateOffsetsWGE16(gtid.x, offsets, keys);
            GroupMemoryBarrierWithGroupSync();
            
            if (gtid.x < RADIX)
                exclusiveHistReduction = g_d[gtid.x]; //take advantage of barrier to grab value
            InitializeLookbackFallback(gtid.x); //Set the locks for lookback fallback in shared memory
            GroupMemoryBarrierWithGroupSync();
        }
    
        if (WaveGetLaneCount() < 16)
        {
            offsets = RankKeysWLT16(gtid.x, keys, SerialIterations());
            
            if (gtid.x < HALF_RADIX)
            {
                uint histReduction = WaveHistInclusiveScanCircularShiftWLT16(gtid.x);
                g_d[gtid.x] = histReduction + (histReduction << 16); //take advantage of barrier to begin scan
                CASDeviceBroadcastReductionsWLT16(gtid.x, partitionIndex, histReduction);
            }
            
            WaveHistReductionExclusiveScanWLT16(gtid.x);
            GroupMemoryBarrierWithGroupSync();
            
            UpdateOffsetsWLT16(gtid.x, SerialIterations(), offsets, keys);
            GroupMemoryBarrierWithGroupSync();
            
            if (gtid.x < RADIX)                 //take advantage of barrier to grab value
                exclusiveHistReduction = g_d[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
            InitializeLookbackFallback(gtid.x); //Set the locks for lookback fallback in shared memory
            GroupMemoryBarrierWithGroupSync();
        }
    
        LookbackWithFallback(gtid.x, partitionIndex, exclusiveHistReduction);
        GroupMemoryBarrierWithGroupSync();
        
        ScatterKeysShared(offsets, keys);
        GroupMemoryBarrierWithGroupSync();
        
        if (partitionIndex < e_threadBlocks - 1)
            ScatterDevice(gtid.x, partitionIndex, offsets);
        
        if (partitionIndex == e_threadBlocks - 1)
            ScatterDevicePartial(gtid.x, partitionIndex, offsets);
    }
}