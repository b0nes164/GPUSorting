/******************************************************************************
 * FFXParllelSort
 * This algorithm is part of the FidelityFX SDK.
 * https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
 * 
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ******************************************************************************/

#define FFX_PARALLELSORT_SORT_BITS_PER_PASS		    4
#define	FFX_PARALLELSORT_SORT_BIN_COUNT			    (1 << FFX_PARALLELSORT_SORT_BITS_PER_PASS)
#define FFX_PARALLELSORT_ELEMENTS_PER_THREAD	    4
#define FFX_PARALLELSORT_THREADGROUP_SIZE		    256

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_numThreadGroups;
    uint e_numBlocksPerThreadGroup;
    uint e_numThreadGroupsWithAdditionalBlocks;
    uint e_numReduceThreadgroupPerBin;
    uint e_numScanValues;
    uint e_radixShift;
    uint padding;
}

RWStructuredBuffer<uint>   SrcBuffer          : register(u0);
RWStructuredBuffer<uint>   SrcPayload         : register(u1);
RWStructuredBuffer<uint>   SumTable           : register(u2);
RWStructuredBuffer<uint>   ReduceTable        : register(u3);
RWStructuredBuffer<uint>   DstBuffer          : register(u4);
RWStructuredBuffer<uint>   DstPayload         : register(u5);
RWStructuredBuffer<uint>   ScanSrc            : register(u6);
RWStructuredBuffer<uint>   ScanDst            : register(u7);
RWStructuredBuffer<uint>   ScanScratch        : register(u8);

groupshared uint gs_FFX_PARALLELSORT_Histogram[FFX_PARALLELSORT_THREADGROUP_SIZE * FFX_PARALLELSORT_SORT_BIN_COUNT];
groupshared uint gs_FFX_PARALLELSORT_LDSSums[FFX_PARALLELSORT_THREADGROUP_SIZE];
groupshared uint gs_FFX_PARALLELSORT_LDS[FFX_PARALLELSORT_ELEMENTS_PER_THREAD][FFX_PARALLELSORT_THREADGROUP_SIZE];
groupshared uint gs_FFX_PARALLELSORT_BinOffsetCache[FFX_PARALLELSORT_THREADGROUP_SIZE];
groupshared uint gs_FFX_PARALLELSORT_LocalHistogram[FFX_PARALLELSORT_SORT_BIN_COUNT];
groupshared uint gs_FFX_PARALLELSORT_LDSScratch[FFX_PARALLELSORT_THREADGROUP_SIZE];

uint FfxNumKeys()
{
    return e_numKeys;
}
int FfxNumBlocksPerThreadGroup()
{
    return e_numBlocksPerThreadGroup;
}
uint FfxNumThreadGroups()
{
    return e_numThreadGroups;
}
uint FfxNumThreadGroupsWithAdditionalBlocks()
{
    return e_numThreadGroupsWithAdditionalBlocks;
}
uint FfxNumReduceThreadgroupPerBin()
{
    return e_numReduceThreadgroupPerBin;
}
uint FfxNumScanValues()
{
    return e_numScanValues;
}
uint FfxShiftBit()
{
    return e_radixShift;
}

uint FfxLoadKey(uint index)
{
    return SrcBuffer[index];
}

void FfxStoreKey(uint index, uint value)
{
    DstBuffer[index] = value;
}

uint FfxLoadPayload(uint index)
{
    return SrcPayload[index];
}

void FfxStorePayload(uint index, uint value)
{
    DstPayload[index] = value;
}

uint FfxLoadSum(uint index)
{
    return SumTable[index];
}

void FfxStoreSum(uint index, uint value)
{
    SumTable[index] = value;
}

void FfxStoreReduce(uint index, uint value)
{
    ReduceTable[index] = value;
}

uint FfxLoadScanSource(uint index)
{
    return ScanSrc[index];
}

void FfxStoreScanDest(uint index, uint value)
{
    ScanDst[index] = value;
}

uint FfxLoadScanScratch(uint index)
{
    return ScanScratch[index];
}

void ffxParallelSortCountUInt(uint localID, uint groupID, uint ShiftBit)
{
    for (int i = 0; i < FFX_PARALLELSORT_SORT_BIN_COUNT; i++)
        gs_FFX_PARALLELSORT_Histogram[(i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID] = 0;
    GroupMemoryBarrierWithGroupSync();

    int BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
    uint NumBlocksPerThreadGroup = FfxNumBlocksPerThreadGroup();
    uint NumThreadGroups = FfxNumThreadGroups();
    uint NumThreadGroupsWithAdditionalBlocks = FfxNumThreadGroupsWithAdditionalBlocks();
    uint NumKeys = FfxNumKeys();

    uint ThreadgroupBlockStart = (BlockSize * NumBlocksPerThreadGroup * groupID);
    uint NumBlocksToProcess = NumBlocksPerThreadGroup;

    if (groupID >= NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)
    {
        ThreadgroupBlockStart += (groupID - (NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)) * BlockSize;
        NumBlocksToProcess++;
    }

    uint BlockIndex = ThreadgroupBlockStart + localID;
    for (uint BlockCount = 0; BlockCount < NumBlocksToProcess; BlockCount++, BlockIndex += BlockSize)
    {
        uint DataIndex = BlockIndex;
        uint srcKeys[FFX_PARALLELSORT_ELEMENTS_PER_THREAD];
        srcKeys[0] = FfxLoadKey(DataIndex);
        srcKeys[1] = FfxLoadKey(DataIndex + FFX_PARALLELSORT_THREADGROUP_SIZE);
        srcKeys[2] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 2));
        srcKeys[3] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 3));

        for (uint i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        {
            if (DataIndex < NumKeys)
            {
                uint localKey = (srcKeys[i] >> ShiftBit) & 0xf;
                InterlockedAdd(gs_FFX_PARALLELSORT_Histogram[(localKey * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID], 1);
                DataIndex += FFX_PARALLELSORT_THREADGROUP_SIZE;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
    {
        uint sum = 0;
        for (int i = 0; i < FFX_PARALLELSORT_THREADGROUP_SIZE; i++)
        {
            sum += gs_FFX_PARALLELSORT_Histogram[localID * FFX_PARALLELSORT_THREADGROUP_SIZE + i];
        }
        FfxStoreSum(localID * NumThreadGroups + groupID, sum);
    }
}

uint ffxParallelSortThreadgroupReduce(uint localSum, uint localID)
{
    uint waveReduced = WaveActiveSum(localSum);
        
    uint waveID = localID / WaveGetLaneCount();
    if (WaveIsFirstLane())
        gs_FFX_PARALLELSORT_LDSSums[waveID] = waveReduced;

    GroupMemoryBarrierWithGroupSync();

    if (!waveID)
        waveReduced = WaveActiveSum((localID < FFX_PARALLELSORT_THREADGROUP_SIZE / WaveGetLaneCount()) ? gs_FFX_PARALLELSORT_LDSSums[localID] : 0);
    
    return waveReduced;
}

uint ffxParallelSortBlockScanPrefix(uint localSum, uint localID)
{
    uint wavePrefixed = WavePrefixSum(localSum);
    uint waveID = localID / WaveGetLaneCount();
    uint laneID = WaveGetLaneIndex();

    if (laneID == WaveGetLaneCount() - 1)
        gs_FFX_PARALLELSORT_LDSSums[waveID] = wavePrefixed + localSum;
    GroupMemoryBarrierWithGroupSync();

    if (!waveID)
        gs_FFX_PARALLELSORT_LDSSums[localID] = WavePrefixSum(gs_FFX_PARALLELSORT_LDSSums[localID]);
    GroupMemoryBarrierWithGroupSync();

    wavePrefixed += gs_FFX_PARALLELSORT_LDSSums[waveID];
    return wavePrefixed;
}

void ffxParallelSortScanPrefix(
    uint numValuesToScan,
    uint localID,
    uint groupID,
    uint BinOffset,
    uint BaseIndex,
    bool AddPartialSums)
{
    for (uint i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        uint DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;

        uint col = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) / FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        uint row = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) % FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        gs_FFX_PARALLELSORT_LDS[row][col] = (DataIndex < numValuesToScan) ? FfxLoadScanSource(BinOffset + DataIndex) : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint threadgroupSum = 0;
    for (uint i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        uint tmp = gs_FFX_PARALLELSORT_LDS[i][localID];
        gs_FFX_PARALLELSORT_LDS[i][localID] = threadgroupSum;
        threadgroupSum += tmp;
    }
    threadgroupSum = ffxParallelSortBlockScanPrefix(threadgroupSum, localID);

    uint partialSum = 0;
    if (AddPartialSums)
    {
        partialSum = FfxLoadScanScratch(groupID);
    }

    for (uint i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        gs_FFX_PARALLELSORT_LDS[i][localID] += threadgroupSum;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        uint DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;

        uint col = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) / FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        uint row = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) % FFX_PARALLELSORT_ELEMENTS_PER_THREAD;

        if (DataIndex < numValuesToScan)
            FfxStoreScanDest(BinOffset + DataIndex, gs_FFX_PARALLELSORT_LDS[row][col] + partialSum);
    }
}

void ffxParallelSortScatterUInt(uint localID, uint groupID, uint ShiftBit)
{
    uint NumBlocksPerThreadGroup = FfxNumBlocksPerThreadGroup();
    uint NumThreadGroups = FfxNumThreadGroups();
    uint NumThreadGroupsWithAdditionalBlocks = FfxNumThreadGroupsWithAdditionalBlocks();
    uint NumKeys = FfxNumKeys();

    if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
        gs_FFX_PARALLELSORT_BinOffsetCache[localID] = FfxLoadSum(localID * NumThreadGroups + groupID);
    GroupMemoryBarrierWithGroupSync();

    int BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
    uint ThreadgroupBlockStart = (BlockSize * NumBlocksPerThreadGroup * groupID);
    uint NumBlocksToProcess = NumBlocksPerThreadGroup;

    if (groupID >= NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)
    {
        ThreadgroupBlockStart += (groupID - (NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)) * BlockSize;
        NumBlocksToProcess++;
    }

    uint BlockIndex = ThreadgroupBlockStart + localID;
    uint newCount;
    for (int BlockCount = 0; BlockCount < NumBlocksToProcess; BlockCount++, BlockIndex += BlockSize)
    {
        uint DataIndex = BlockIndex;

        uint srcKeys[FFX_PARALLELSORT_ELEMENTS_PER_THREAD];
        srcKeys[0] = FfxLoadKey(DataIndex);
        srcKeys[1] = FfxLoadKey(DataIndex + FFX_PARALLELSORT_THREADGROUP_SIZE);
        srcKeys[2] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 2));
        srcKeys[3] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 3));

#ifdef FFX_PARALLELSORT_COPY_VALUE
        uint srcValues[FFX_PARALLELSORT_ELEMENTS_PER_THREAD];
        srcValues[0] = FfxLoadPayload(DataIndex);
        srcValues[1] = FfxLoadPayload(DataIndex + FFX_PARALLELSORT_THREADGROUP_SIZE);
        srcValues[2] = FfxLoadPayload(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 2));
        srcValues[3] = FfxLoadPayload(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 3));
#endif

        for (int i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        {
            if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
                gs_FFX_PARALLELSORT_LocalHistogram[localID] = 0;

            uint localKey = (DataIndex < NumKeys ? srcKeys[i] : 0xffffffff);
#ifdef FFX_PARALLELSORT_COPY_VALUE
            uint localValue = (DataIndex < NumKeys ? srcValues[i] : 0);
#endif 

            for (uint bitShift = 0; bitShift < FFX_PARALLELSORT_SORT_BITS_PER_PASS; bitShift += 2)
            {
                uint keyIndex = (localKey >> ShiftBit) & 0xf;
                uint bitKey = (keyIndex >> bitShift) & 0x3;
                uint packedHistogram = 1 << (bitKey * 8);
                uint localSum = ffxParallelSortBlockScanPrefix(packedHistogram, localID);
                if (localID == (FFX_PARALLELSORT_THREADGROUP_SIZE - 1))
                    gs_FFX_PARALLELSORT_LDSScratch[0] = localSum + packedHistogram;
                GroupMemoryBarrierWithGroupSync();

                packedHistogram = gs_FFX_PARALLELSORT_LDSScratch[0];
                packedHistogram = (packedHistogram << 8) + (packedHistogram << 16) + (packedHistogram << 24);
                localSum += packedHistogram;

                uint keyOffset = (localSum >> (bitKey * 8)) & 0xff;

                gs_FFX_PARALLELSORT_LDSSums[keyOffset] = localKey;
                GroupMemoryBarrierWithGroupSync();
                
                localKey = gs_FFX_PARALLELSORT_LDSSums[localID];
                GroupMemoryBarrierWithGroupSync();

#ifdef FFX_PARALLELSORT_COPY_VALUE
                    gs_FFX_PARALLELSORT_LDSSums[keyOffset] = localValue;
                    GroupMemoryBarrierWithGroupSync();
                    localValue = gs_FFX_PARALLELSORT_LDSSums[localID];
                    GroupMemoryBarrierWithGroupSync();
#endif
            }
            
            uint keyIndex = (localKey >> ShiftBit) & 0xf;
            InterlockedAdd(gs_FFX_PARALLELSORT_LocalHistogram[keyIndex], 1);
            GroupMemoryBarrierWithGroupSync();


            uint histogramPrefixSum = WavePrefixSum(localID < FFX_PARALLELSORT_SORT_BIN_COUNT ? gs_FFX_PARALLELSORT_LocalHistogram[localID] : 0);
            if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
                gs_FFX_PARALLELSORT_LDSScratch[localID] = histogramPrefixSum;

            uint globalOffset = gs_FFX_PARALLELSORT_BinOffsetCache[keyIndex];
            GroupMemoryBarrierWithGroupSync();

            uint localOffset = localID - gs_FFX_PARALLELSORT_LDSScratch[keyIndex];
            uint totalOffset = globalOffset + localOffset;

            if (totalOffset < NumKeys)
            {
                FfxStoreKey(totalOffset, localKey);

#ifdef FFX_PARALLELSORT_COPY_VALUE
                FfxStorePayload(totalOffset, localValue);
#endif
            }
            GroupMemoryBarrierWithGroupSync();

            if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
                gs_FFX_PARALLELSORT_BinOffsetCache[localID] += gs_FFX_PARALLELSORT_LocalHistogram[localID];

            DataIndex += FFX_PARALLELSORT_THREADGROUP_SIZE;
        }
    }
}

[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void FPS_Count(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    ffxParallelSortCountUInt(localID, groupID, FfxShiftBit());
}

[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void FPS_CountReduce(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    uint NumReduceThreadgroupPerBin = FfxNumReduceThreadgroupPerBin();
    uint NumThreadGroups = FfxNumThreadGroups();

    uint BinID = groupID / NumReduceThreadgroupPerBin;
    uint BinOffset = BinID * NumThreadGroups;

    uint BaseIndex = (groupID % NumReduceThreadgroupPerBin) * FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;

    uint threadgroupSum = 0;
    for (uint i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; ++i)
    {
        uint DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;
        threadgroupSum += (DataIndex < NumThreadGroups) ? FfxLoadSum(BinOffset + DataIndex) : 0;
    }

    threadgroupSum = ffxParallelSortThreadgroupReduce(threadgroupSum, localID);

    if (localID == 0)
        FfxStoreReduce(groupID, threadgroupSum);
}

[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void FPS_Scan(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
   uint BaseIndex = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE * groupID;
   ffxParallelSortScanPrefix(FfxNumScanValues(), localID, groupID, 0, BaseIndex, false);
}

[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void FPS_ScanAdd(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    uint BinID = groupID / FfxNumReduceThreadgroupPerBin();
    uint BinOffset = BinID * FfxNumThreadGroups();
    uint BaseIndex = (groupID % FfxNumReduceThreadgroupPerBin()) * FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
    ffxParallelSortScanPrefix(FfxNumThreadGroups(), localID, groupID, BinOffset, BaseIndex, true);
}

[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void FPS_Scatter(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    ffxParallelSortScatterUInt(localID, groupID, FfxShiftBit());
}