/******************************************************************************
 * GPUSorting
 * OneSweep Implementation
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 * 
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 ******************************************************************************/
//Compiler Defines
//#define KEY_UINT KEY_INT KEY_FLOAT
//#define PAYLOAD_UINT PAYLOAD_INT PAYLOAD_FLOAT
//#define SHOULD_ASCEND
//#define SORT_PAIRS
//#define ENABLE_16_BIT
//#define LOCK_TO_W32               //Used to lock RDNA to 32, we want WGP's not CU's
#include "SweepCommon.hlsl"

//DigitBinningPass does not require flattening
//Lock RDNA to 32, we want WGP's not CU's
#if defined(LOCK_TO_W32)
[WaveSize(32)]
#endif
[numthreads(D_DIM, 1, 1)]
void DigitBinningPass(uint3 gtid : SV_GroupThreadID)
{
    uint partitionIndex;
    KeyStruct keys;
    OffsetStruct offsets;
    
    //WGT 16 can potentially skip some barriers
    if (WaveGetLaneCount() > 16)
    {
        if(WaveHistsSizeWGE16() < PART_SIZE)
            ClearWaveHists(gtid.x);

        AssignPartitionTile(gtid.x, partitionIndex);
        if(WaveHistsSizeWGE16() >= PART_SIZE)
        {
            GroupMemoryBarrierWithGroupSync();
            ClearWaveHists(gtid.x);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    
    if (WaveGetLaneCount() <= 16)
    {
        AssignPartitionTile(gtid.x, partitionIndex);
        GroupMemoryBarrierWithGroupSync();
        ClearWaveHists(gtid.x);
        GroupMemoryBarrierWithGroupSync();
    }
    
    if(partitionIndex < e_threadBlocks - 1)
    {
        if (WaveGetLaneCount() >= 16)
            keys = LoadKeysWGE16(gtid.x, partitionIndex);
        
        if (WaveGetLaneCount() < 16)
            keys = LoadKeysWLT16(gtid.x, partitionIndex, SerialIterations());
    }
        
    if(partitionIndex == e_threadBlocks - 1)
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
            DeviceBroadcastReductionsWGE16(gtid.x, partitionIndex, histReduction);
            histReduction += WavePrefixSum(histReduction); //take advantage of barrier to begin scan
        }
        GroupMemoryBarrierWithGroupSync();

        WaveHistReductionExclusiveScanWGE16(gtid.x, histReduction);
        GroupMemoryBarrierWithGroupSync();
            
        UpdateOffsetsWGE16(gtid.x, offsets, keys);
        if (gtid.x < RADIX)
            exclusiveHistReduction = g_d[gtid.x]; //take advantage of barrier to grab value
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (WaveGetLaneCount() < 16)
    {
        offsets = RankKeysWLT16(gtid.x, keys, SerialIterations());
            
        if (gtid.x < HALF_RADIX)
        {
            uint histReduction = WaveHistInclusiveScanCircularShiftWLT16(gtid.x);
            g_d[gtid.x] = histReduction + (histReduction << 16); //take advantage of barrier to begin scan
            DeviceBroadcastReductionsWLT16(gtid.x, partitionIndex, histReduction);
        }
            
        WaveHistReductionExclusiveScanWLT16(gtid.x);
        GroupMemoryBarrierWithGroupSync();
            
        UpdateOffsetsWLT16(gtid.x, SerialIterations(), offsets, keys);
        if (gtid.x < RADIX) //take advantage of barrier to grab value
            exclusiveHistReduction = g_d[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
        GroupMemoryBarrierWithGroupSync();
    }
    
    ScatterKeysShared(offsets, keys);
    Lookback(gtid.x, partitionIndex, exclusiveHistReduction);
    GroupMemoryBarrierWithGroupSync();
    
    if(partitionIndex < e_threadBlocks - 1)
        ScatterDevice(gtid.x, partitionIndex, offsets);
        
    if(partitionIndex == e_threadBlocks - 1)
        ScatterDevicePartial(gtid.x, partitionIndex, offsets);
}