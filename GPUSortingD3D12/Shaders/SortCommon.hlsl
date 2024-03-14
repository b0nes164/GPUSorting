/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 3/13/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
//Compiler Defines
//#define KEY_UINT KEY_INT KEY_FLOAT
//#define PAYLOAD_UINT PAYLOAD_INT PAYLOAD_FLOAT
//#define SHOULD_ASCEND
//#define SORT_PAIRS
//#define ENABLE_16_BIT

//General macros
#if defined(SORT_PAIRS)
#define PART_SIZE       7680U   //size of a partition tile
#define PASS_DIM        512U    //The number of threads int digit binning pass
#else
#define PART_SIZE       3840U
#define PASS_DIM        256U
#endif

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
#if defined(SORT_PAIRS)
#define MAX_PASS_SMEM   8192U       //shared memory for DigitBinningPass kernel
#else
#define MAX_PASS_SMEM   4096U       //shared memory for DigitBinningPass kernel
#endif

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_radixShift;
    uint e_threadBlocks;
    uint padding;
};

#if defined(KEY_UINT)
RWStructuredBuffer<uint> b_sort         : register(u0);
RWStructuredBuffer<uint> b_alt          : register(u1);
#elif defined(KEY_INT)
RWStructuredBuffer<int> b_sort          : register(u0);
RWStructuredBuffer<int> b_alt           : register(u1);
#elif defined(KEY_FLOAT)
RWStructuredBuffer<float> b_sort        : register(u0);
RWStructuredBuffer<float> b_alt         : register(u1);
#endif

#if defined(PAYLOAD_UINT)
RWStructuredBuffer<uint> b_sortPayload  : register(u2);
RWStructuredBuffer<uint> b_altPayload   : register(u3);
#elif defined(PAYLOAD_INT)
RWStructuredBuffer<int> b_sortPayload   : register(u2);
RWStructuredBuffer<int> b_altPayload    : register(u3);
#elif defined(PAYLOAD_FLOAT)
RWStructuredBuffer<float> b_sortPayload : register(u2);
RWStructuredBuffer<float> b_altPayload  : register(u3);
#endif

groupshared uint g_pass[MAX_PASS_SMEM]; //Shared memory for DigitBinningPass

struct KeyStruct
{
    uint k[KEYS_PER_THREAD];
};

struct OffsetStruct
{
#if defined(ENABLE_16_BIT)
    uint16_t o[KEYS_PER_THREAD];
#else
    uint o[KEYS_PER_THREAD];
#endif
};

//*****************************************************************************
//HELPER FUNCTIONS
//*****************************************************************************
inline uint getWaveIndex(uint gtid)
{
    return gtid / WaveGetLaneCount();
}

//Radix Tricks by Michael Herf
//http://stereopsis.com/radix.html
inline uint FloatToUint(float f)
{
    uint mask = -((int) (asuint(f) >> 31)) | 0x80000000;
    return asuint(f) ^ mask;
}

inline float UintToFloat(uint u)
{
    uint mask = ((u >> 31) - 1) | 0x80000000;
    return asfloat(u ^ mask);
}

inline uint IntToUint(int i)
{
    return asuint(i ^ 0x80000000);
}

inline int UintToInt(uint u)
{
    return asint(u ^ 0x80000000);
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

inline uint DeviceOffsetWGE16(uint gtid, uint partIndex)
{
    return SharedOffsetWGE16(gtid) + partIndex * PART_SIZE;
}

inline uint DeviceOffsetWLT16(uint gtid, uint partIndex, uint serialIterations)
{
    return SharedOffsetWLT16(gtid, serialIterations) + partIndex * PART_SIZE;
}

inline uint GlobalHistOffset()
{
    return e_radixShift << 5;
}

inline uint WaveHistsSizeWGE16()
{
    return PASS_DIM / WaveGetLaneCount() * RADIX;
}

inline uint WaveHistsSizeWLT16()
{
    return MAX_PASS_SMEM;
}

//*****************************************************************************
//FUNCTIONS COMMON TO THE DOWNSWEEP / DIGIT BINNING PASS
//*****************************************************************************
//If the size of  a wave is too small, we do not have enough space in
//shared memory to assign a histogram to each wave, so instead,
//some operations are peformed serially.
inline uint SerialIterations()
{
    return (PASS_DIM / WaveGetLaneCount() + 31) / 32;
}

inline void LoadKey(inout uint key, uint index)
{
#if defined(KEY_UINT)
    key = b_sort[index];
#elif defined(KEY_INT)
    key = UintToInt(b_sort[index]);
#elif defined(KEY_FLOAT)
    key = FloatToUint(b_sort[index]);
#endif
}

inline void LoadDummyKey(inout uint key)
{
    key = 0xffffffff;
}

inline KeyStruct LoadKeysWGE16(uint gtid, uint partIndex)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, partIndex);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount())
    {
        LoadKey(keys.k[i], t);
    }
    return keys;
}

inline KeyStruct LoadKeysWLT16(uint gtid, uint partIndex, uint serialIterations)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        LoadKey(keys.k[i], t);
    }
    return keys;
}

inline KeyStruct LoadKeysPartialWGE16(uint gtid, uint partIndex)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, partIndex);
                 i < KEYS_PER_THREAD;
                 ++i, t += WaveGetLaneCount())
    {
        if (t < e_numKeys)
            LoadKey(keys.k[i], t);
        else
            LoadDummyKey(keys.k[i]);
    }
    return keys;
}

inline KeyStruct LoadKeysPartialWLT16(uint gtid, uint partIndex, uint serialIterations)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        if (t < e_numKeys)
            LoadKey(keys.k[i], t);
        else
            LoadDummyKey(keys.k[i]);
    }
    return keys;
}

inline uint WaveFlagsWGE16()
{
    return (WaveGetLaneCount() & 31) ?
        (1U << WaveGetLaneCount()) - 1 : 0xffffffff;
}

inline uint WaveFlagsWLT16()
{
    return (1U << WaveGetLaneCount()) - 1;;
}

inline void WarpLevelMultiSplitWGE16(uint key, uint waveParts, inout uint4 waveFlags)
{
    [unroll]
    for (uint k = 0; k < RADIX_LOG; ++k)
    {
        const bool t = key >> (k + e_radixShift) & 1;
        const uint4 ballot = WaveActiveBallot(t);
        for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
            waveFlags[wavePart] &= (t ? 0 : 0xffffffff) ^ ballot[wavePart];
    }
}

inline void WarpLevelMultiSplitWLT16(uint key, inout uint waveFlags)
{
    [unroll]
    for (uint k = 0; k < RADIX_LOG; ++k)
    {
        const bool t = key >> (k + e_radixShift) & 1;
        waveFlags &= (t ? 0 : 0xffffffff) ^ (uint) WaveActiveBallot(t);
    }
}

inline void CountPeerBits(
    inout uint peerBits,
    inout uint totalBits,
    uint4 waveFlags,
    uint waveParts)
{
    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
    {
        if (WaveGetLaneIndex() >= wavePart * 32)
        {
            const uint ltMask = WaveGetLaneIndex() >= (wavePart + 1) * 32 ?
                0xffffffff : (1U << (WaveGetLaneIndex() & 31)) - 1;
            peerBits += countbits(waveFlags[wavePart] & ltMask);
        }
        totalBits += countbits(waveFlags[wavePart]);
    }
}

inline uint CountPeerBitsWLT16(
    uint waveFlags,
    uint ltMask)
{
    return countbits(waveFlags & ltMask);
}

inline uint FindLowestRankPeer(
    uint4 waveFlags,
    uint waveParts)
{
    uint lowestRankPeer = 0;
    for (uint wavePart = 0; wavePart < waveParts; ++wavePart)
    {
        uint fbl = firstbitlow(waveFlags[wavePart]);
        if (fbl == 0xffffffff)
            lowestRankPeer += 32;
        else
            return lowestRankPeer + fbl;
    }
    return 0; //will never happen
}

inline OffsetStruct RankKeysWGE16(uint gtid, KeyStruct keys)
{
    OffsetStruct offsets;
    const uint waveParts = (WaveGetLaneCount() + 31) / 32;
    [unroll]
    for (uint i = 0; i < KEYS_PER_THREAD; ++i)
    {
        uint4 waveFlags = WaveFlagsWGE16();
        WarpLevelMultiSplitWGE16(keys.k[i], waveParts, waveFlags);
        
        const uint index = ExtractDigit(keys.k[i]) + (getWaveIndex(gtid.x) * RADIX);
        const uint lowestRankPeer = FindLowestRankPeer(waveFlags, waveParts);
        
        uint peerBits = 0;
        uint totalBits = 0;
        CountPeerBits(peerBits, totalBits, waveFlags, waveParts);
        
        uint preIncrementVal;
        if (peerBits == 0)
            InterlockedAdd(g_pass[index], totalBits, preIncrementVal);
        offsets.o[i] = WaveReadLaneAt(preIncrementVal, lowestRankPeer) + peerBits;
    }
    
    return offsets;
}

inline OffsetStruct RankKeysWLT16(uint gtid, KeyStruct keys, uint serialIterations)
{
    OffsetStruct offsets;
    const uint ltMask = (1U << WaveGetLaneIndex()) - 1;
    
    [unroll]
    for (uint i = 0; i < KEYS_PER_THREAD; ++i)
    {
        uint waveFlags = WaveFlagsWLT16();
        WarpLevelMultiSplitWLT16(keys.k[i], waveFlags);
        
        const uint index = ExtractPackedIndex(keys.k[i]) +
                (getWaveIndex(gtid.x) / serialIterations * HALF_RADIX);
        
        const uint peerBits = CountPeerBitsWLT16(waveFlags, ltMask);
        for (uint k = 0; k < serialIterations; ++k)
        {
            if (getWaveIndex(gtid.x) % serialIterations == k)
                offsets.o[i] = ExtractPackedValue(g_pass[index], keys.k[i]) + peerBits;
            
            //shuffle trick will not work, because potentially two threads will
            //atomically add to same digit counter due to bit packing
            GroupMemoryBarrierWithGroupSync();
            if (getWaveIndex(gtid.x) % serialIterations == k && peerBits == 0)
            {
                InterlockedAdd(g_pass[index],
                    countbits(waveFlags) << ExtractPackedShift(keys.k[i]));
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    
    return offsets;
}

inline uint WaveHistInclusiveScanCircularShiftWGE16(uint gtid)
{
    uint histReduction = g_pass[gtid];
    for (uint i = gtid + RADIX; i < WaveHistsSizeWGE16(); i += RADIX)
    {
        histReduction += g_pass[i];
        g_pass[i] = histReduction - g_pass[i];
    }
    return histReduction;
}

inline uint WaveHistInclusiveScanCircularShiftWLT16(uint gtid)
{
    uint histReduction = g_pass[gtid];
    for (uint i = gtid + HALF_RADIX; i < WaveHistsSizeWLT16(); i += HALF_RADIX)
    {
        histReduction += g_pass[i];
        g_pass[i] = histReduction - g_pass[i];
    }
    return histReduction;
}

inline void WaveHistReductionExclusiveScanWGE16(uint gtid, uint histReduction)
{
    if (gtid < RADIX)
    {
        const uint laneMask = WaveGetLaneCount() - 1;
        g_pass[((WaveGetLaneIndex() + 1) & laneMask) + (gtid & ~laneMask)] = histReduction;
    }
    GroupMemoryBarrierWithGroupSync();
                
    if (gtid < RADIX / WaveGetLaneCount())
    {
        g_pass[gtid * WaveGetLaneCount()] =
            WavePrefixSum(g_pass[gtid * WaveGetLaneCount()]);
    }
    GroupMemoryBarrierWithGroupSync();
                
    if (gtid < RADIX && WaveGetLaneIndex())
        g_pass[gtid] += WaveReadLaneAt(g_pass[gtid - 1], 1);
}

//inclusive/exclusive prefix sum up the histograms,
//use a blelloch scan for in place packed exclusive
inline void WaveHistReductionExclusiveScanWLT16(uint gtid)
{
    uint shift = 1;
    for (uint j = RADIX >> 2; j > 0; j >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gtid < j)
        {
            g_pass[((((gtid << 1) + 2) << shift) - 1) >> 1] +=
                g_pass[((((gtid << 1) + 1) << shift) - 1) >> 1] & 0xffff0000;
        }
        shift++;
    }
    GroupMemoryBarrierWithGroupSync();
                
    if (gtid == 0)
        g_pass[HALF_RADIX - 1] &= 0xffff;
                
    for (uint j = 1; j < RADIX >> 1; j <<= 1)
    {
        --shift;
        GroupMemoryBarrierWithGroupSync();
        if (gtid < j)
        {
            const uint t = ((((gtid << 1) + 1) << shift) - 1) >> 1;
            const uint t2 = ((((gtid << 1) + 2) << shift) - 1) >> 1;
            const uint t3 = g_pass[t];
            g_pass[t] = (g_pass[t] & 0xffff) | (g_pass[t2] & 0xffff0000);
            g_pass[t2] += t3 & 0xffff0000;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gtid < HALF_RADIX)
    {
        const uint t = g_pass[gtid];
        g_pass[gtid] = (t >> 16) + (t << 16) + (t & 0xffff0000);
    }
}

inline void UpdateOffsetsWGE16(uint gtid, inout OffsetStruct offsets, KeyStruct keys)
{
    if (gtid >= WaveGetLaneCount())
    {
        const uint t = getWaveIndex(gtid) * RADIX;
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
        {
            const uint t2 = ExtractDigit(keys.k[i]);
            offsets.o[i] += g_pass[t2 + t] + g_pass[t2];
        }
    }
    else
    {
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            offsets.o[i] += g_pass[ExtractDigit(keys.k[i])];
    }
}

inline void UpdateOffsetsWLT16(
    uint gtid,
    uint serialIterations,
    inout OffsetStruct offsets,
    KeyStruct keys)
{
    if (gtid >= WaveGetLaneCount() * serialIterations)
    {
        const uint t = getWaveIndex(gtid) / serialIterations * HALF_RADIX;
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
        {
            const uint t2 = ExtractPackedIndex(keys.k[i]);
            offsets.o[i] += ExtractPackedValue(g_pass[t2 + t] + g_pass[t2], keys.k[i]);
        }
    }
    else
    {
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            offsets.o[i] += ExtractPackedValue(g_pass[ExtractPackedIndex(keys.k[i])], keys.k[i]);
    }
}

inline void ScatterKeysShared(OffsetStruct offsets, KeyStruct keys)
{
    [unroll]
    for (uint i = 0; i < KEYS_PER_THREAD; ++i)
        g_pass[offsets.o[i]] = keys.k[i];
}

inline uint DescendingIndex(uint deviceIndex)
{
    return e_numKeys - deviceIndex - 1;
}

inline void WriteKey(uint deviceIndex, uint groupSharedIndex)
{
#if defined(KEY_UINT)
    b_alt[deviceIndex] = g_pass[groupSharedIndex];
#elif defined(KEY_INT)
    b_alt[deviceIndex] = UintToInt(g_pass[groupSharedIndex]);
#elif defined(KEY_FLOAT)
    b_alt[deviceIndex] = UintToFloat(g_pass[groupSharedIndex]);
#endif
}

inline void LoadPayload(uint deviceIndex, uint groupSharedIndex)
{
#if defined(PAYLOAD_UINT)
    g_pass[groupSharedIndex] = b_sortPayload[deviceIndex];
#elif defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT)
    g_pass[groupSharedIndex] = asuint(b_sortPayload[deviceIndex]);
#endif
}

inline void WritePayload(uint deviceIndex, uint groupSharedIndex)
{
#if defined(PAYLOAD_UINT)
    b_altPayload[deviceIndex] = g_pass[groupSharedIndex];
#elif defined(PAYLOAD_INT)
    b_altPayload[deviceIndex] = asint(g_pass[groupSharedIndex]);
#elif defined(PAYLOAD_FLOAT)
    b_altPayload[deviceIndex] = asfloat(g_pass[groupSharedIndex]);
#endif
}

//*****************************************************************************
//SCATTERING: FULL PARTITIONS
//*****************************************************************************
//KEYS ONLY
inline void ScatterKeysOnlyDeviceAscending(uint gtid)
{
    for (uint i = gtid; i < PART_SIZE; i += PASS_DIM)
        WriteKey(g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i, i);
}

inline void ScatterKeysOnlyDeviceDescending(uint gtid)
{
    if(e_radixShift == 24)
    {
        for (uint i = gtid; i < PART_SIZE; i += PASS_DIM)
            WriteKey(DescendingIndex(g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i), i);
    }
    else
    {
        ScatterKeysOnlyDeviceAscending(gtid);
    }
}

inline void ScatterKeysOnlyDevice(uint gtid)
{
#if defined(SHOULD_ASCEND)
    ScatterKeysOnlyDeviceAscending(gtid);
#else
    ScatterKeysOnlyDeviceDescending(gtid);
#endif
}

//KEY VALUE PAIRS
inline void ScatterPairsKeyPhaseAscendingWGE16(uint gtid, inout KeyStruct keys)
{
    [unroll]
    for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount())
    {
        keys.k[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
        WriteKey(keys.k[i], t);
    }
}

inline void ScatterPairsKeyPhaseAscendingWLT16(
    uint gtid,
    uint serialIterations,
    inout KeyStruct keys)
{
    [unroll]
    for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        keys.k[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
        WriteKey(keys.k[i], t);
    }
}

inline void ScatterPairsKeyPhaseDescendingWGE16(uint gtid, inout KeyStruct keys)
{
    if (e_radixShift == 24)
    {
        [unroll]
        for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount())
        {
            keys.k[i] = DescendingIndex(g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t);
            WriteKey(keys.k[i], t);
        }
    }
    else
    {
        ScatterPairsKeyPhaseAscendingWGE16(gtid, keys);
    }
}

inline void ScatterPairsKeyPhaseDescendingWLT16(
    uint gtid,
    uint serialIterations,
    inout KeyStruct keys)
{
    if (e_radixShift == 24)
    {
        [unroll]
        for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount())
        {
            keys.k[i] = DescendingIndex(g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t);
            WriteKey(keys.k[i], t);
        }
    }
    else
    {
        ScatterPairsKeyPhaseAscendingWLT16(gtid, serialIterations, keys);
    }
}

inline void LoadPayloadsWGE16(
    uint gtid,
    uint partIndex,
    OffsetStruct offsets)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, partIndex);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount())
    {
        LoadPayload(t, offsets.o[i]);
    }
}

inline void LoadPayloadsWLT16(
    uint gtid,
    uint partIndex,
    uint serialIterations,
    OffsetStruct offsets)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        LoadPayload(t, offsets.o[i]);
    }
}

inline void ScatterPayloadsWGE16(uint gtid, KeyStruct keys)
{
    [unroll]
    for (uint i = 0, t = SharedOffsetWGE16(gtid);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount())
    {
        WritePayload(keys.k[i], t);
    }
}

inline void ScatterPayloadsWLT16(
    uint gtid,
    uint serialIterations,
    KeyStruct keys)
{
    [unroll]
    for (uint i = 0, t = SharedOffsetWLT16(gtid, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        WritePayload(keys.k[i], t);
    }
}

inline void ScatterPairsDeviceWGE16(
    uint gtid,
    uint partIndex,
    KeyStruct keys,
    OffsetStruct offsets)
{
#if defined(SHOULD_ASCEND)
    ScatterPairsKeyPhaseAscendingWGE16(gtid, keys);
#else
    ScatterPairsKeyPhaseDescendingWGE16(gtid, keys);
#endif
    GroupMemoryBarrierWithGroupSync();
    LoadPayloadsWGE16(gtid, partIndex, offsets);
    GroupMemoryBarrierWithGroupSync();
    ScatterPayloadsWGE16(gtid, keys);
}

inline void ScatterPairsDeviceWLT16(
    uint gtid,
    uint partIndex,
    uint serialIterations,
    KeyStruct keys,
    OffsetStruct offsets)
{
#if defined(SHOULD_ASCEND)
    ScatterPairsKeyPhaseAscendingWLT16(gtid, serialIterations, keys);
#else
    ScatterPairsKeyPhaseDescendingWLT16(gtid, serialIterations, keys);
#endif
    GroupMemoryBarrierWithGroupSync();
    LoadPayloadsWLT16(gtid, partIndex, serialIterations, offsets);
    GroupMemoryBarrierWithGroupSync();
    ScatterPayloadsWLT16(gtid, serialIterations, keys);
}

inline void ScatterDeviceWGE16(
    uint gtid,
    uint partIndex,
    KeyStruct keys,
    OffsetStruct offsets)
{
#if defined(SORT_PAIRS)
    ScatterPairsDeviceWGE16(
        gtid,
        partIndex,
        keys,
        offsets);
#else
    ScatterKeysOnlyDevice(gtid);
#endif
}

inline void ScatterDeviceWLT16(
    uint gtid,
    uint partIndex,
    uint serialIterations,
    KeyStruct keys,
    OffsetStruct offsets)
{
#if defined(SORT_PAIRS)
    ScatterPairsDeviceWLT16(
        gtid,
        partIndex,
        serialIterations,
        keys,
        offsets);
#else
    ScatterKeysOnlyDevice(gtid);
#endif
}

//*****************************************************************************
//SCATTERING: PARTIAL PARTITIONS
//*****************************************************************************
//Determine how many keys a thread will process in a partial partition
inline uint FinalKey(
    uint gtid,
    uint partIndex,
    uint serialIterations)
{
    int subPartSize = (int)(e_numKeys - partIndex * PART_SIZE) -
            (int)(getWaveIndex(gtid) / serialIterations * KEYS_PER_THREAD * WaveGetLaneCount() * serialIterations);
    if (subPartSize > 0)
    {
        if ((uint) subPartSize >= KEYS_PER_THREAD * WaveGetLaneCount() * serialIterations)
        {
            return KEYS_PER_THREAD;
        }
        else
        {
            uint finalKey = subPartSize / WaveGetLaneCount() / serialIterations;
            subPartSize -= finalKey * WaveGetLaneCount() * serialIterations;
            if (WaveGetLaneIndex() + getWaveIndex(gtid) % serialIterations * WaveGetLaneCount() < (uint)subPartSize)
                finalKey++;
            return finalKey;
        }
    }
    else
    {
        return 0;
    }
}

//KEYS ONLY
inline void ScatterKeysOnlyDevicePartialAscending(uint gtid, uint finalPartSize)
{
    for (uint i = gtid; i < finalPartSize; i += PASS_DIM)
        WriteKey(g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i, i);
}

inline void ScatterKeysOnlyDevicePartialDescending(uint gtid, uint finalPartSize)
{
    if (e_radixShift == 24)
    {
        for (uint i = gtid; i < finalPartSize; i += PASS_DIM)
            WriteKey(DescendingIndex(g_pass[ExtractDigit(g_pass[i]) + PART_SIZE] + i), i);
    }
    else
    {
        ScatterKeysOnlyDevicePartialAscending(gtid, finalPartSize);
    }
}

inline void ScatterKeysOnlyDevicePartial(uint gtid, uint partIndex)
{
    const uint finalPartSize = e_numKeys - partIndex * PART_SIZE;
#if defined(SHOULD_ASCEND)
    ScatterKeysOnlyDevicePartialAscending(gtid, finalPartSize);
#else
    ScatterKeysOnlyDevicePartialDescending(gtid, finalPartSize);
#endif
}

inline void ScatterPairsKeyPhaseAscendingPartialWGE16(
    uint gtid,
    uint finalKey,
    inout KeyStruct keys)
{
    for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
        i < finalKey;
        ++i, t += WaveGetLaneCount())
    {
        keys.k[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
        WriteKey(keys.k[i], t);
    }
}

//KEY VALUE PAIRS
inline void ScatterPairsKeyPhaseAscendingPartialWLT16(
    uint gtid,
    uint finalKey,
    uint serialIterations,
    inout KeyStruct keys)
{
    for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
        i < finalKey;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        keys.k[i] = g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t;
        WriteKey(keys.k[i], t);
    }
}

inline void ScatterPairsKeyPhaseDescendingPartialWGE16(
    uint gtid,
    uint finalKey,
    inout KeyStruct keys)
{
    if (e_radixShift == 24)
    {
        for (uint i = 0, t = SharedOffsetWGE16(gtid.x);
        i < finalKey;
        ++i, t += WaveGetLaneCount())
        {
            keys.k[i] = DescendingIndex(g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t);
            WriteKey(keys.k[i], t);
        }
    }
    else
    {
        ScatterPairsKeyPhaseAscendingPartialWGE16(gtid, finalKey, keys);
    }
}

inline void ScatterPairsKeyPhaseDescendingPartialWLT16(
    uint gtid,
    uint finalKey,
    uint serialIterations,
    inout KeyStruct keys)
{
    if (e_radixShift == 24)
    {
        for (uint i = 0, t = SharedOffsetWLT16(gtid.x, serialIterations);
        i < finalKey;
        ++i, t += WaveGetLaneCount())
        {
            keys.k[i] = DescendingIndex(g_pass[ExtractDigit(g_pass[t]) + PART_SIZE] + t);
            WriteKey(keys.k[i], t);
        }
    }
    else
    {
        ScatterPairsKeyPhaseAscendingPartialWLT16(gtid, finalKey, serialIterations, keys);
    }
}

inline void LoadPayloadsPartialWGE16(
    uint gtid,
    uint finalKey,
    uint partIndex,
    OffsetStruct offsets)
{
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, partIndex);
        i < finalKey;
        ++i, t += WaveGetLaneCount())
    {
        LoadPayload(t, offsets.o[i]);
    }
}

inline void LoadPayloadsPartialWLT16(
    uint gtid,
    uint finalKey,
    uint partIndex,
    uint serialIterations,
    OffsetStruct offsets)
{
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, partIndex, serialIterations);
        i < finalKey;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        LoadPayload(t, offsets.o[i]);
    }
}

inline void ScatterPayloadsPartialWGE16(
    uint gtid,
    uint finalKey,
    KeyStruct keys)
{
    for (uint i = 0, t = SharedOffsetWGE16(gtid);
        i < finalKey;
        ++i, t += WaveGetLaneCount())
    {
        WritePayload(keys.k[i], t);
    }
}

inline void ScatterPayloadsPartialWLT16(
    uint gtid,
    uint finalKey,
    uint serialIterations,
    KeyStruct keys)
{
    for (uint i = 0, t = SharedOffsetWLT16(gtid, serialIterations);
        i < finalKey;
        ++i, t += WaveGetLaneCount() * serialIterations)
    {
        WritePayload(keys.k[i], t);
    }
}

inline void ScatterPairsDevicePartialWLT16(
    uint gtid,
    uint partIndex,
    uint serialIterations,
    KeyStruct keys,
    OffsetStruct offsets)
{
    const uint finalKey = FinalKey(gtid, partIndex, serialIterations);
#if defined(SHOULD_ASCEND)
    ScatterPairsKeyPhaseAscendingPartialWLT16(gtid, finalKey, serialIterations, keys);
#else
    ScatterPairsKeyPhaseDescendingPartialWLT16(gtid, finalKey, serialIterations, keys);
#endif
    GroupMemoryBarrierWithGroupSync();
    LoadPayloadsPartialWLT16(gtid, finalKey, partIndex, serialIterations, offsets);
    GroupMemoryBarrierWithGroupSync();
    ScatterPayloadsPartialWLT16(gtid, finalKey, serialIterations, keys);
}

inline void ScatterPairsDevicePartialWGE16(
    uint gtid,
    uint partIndex,
    KeyStruct keys,
    OffsetStruct offsets)
{
    const uint finalKey = FinalKey(gtid, partIndex, 1);
#if defined(SHOULD_ASCEND)
    ScatterPairsKeyPhaseAscendingPartialWGE16(gtid, finalKey, keys);
#else
    ScatterPairsKeyPhaseDescendingPartialWGE16(gtid, finalKey, keys);
#endif
    GroupMemoryBarrierWithGroupSync();
    LoadPayloadsPartialWGE16(gtid, finalKey, partIndex, offsets);
    GroupMemoryBarrierWithGroupSync();
    ScatterPayloadsPartialWGE16(gtid, finalKey, keys);
}

inline void ScatterDevicePartialWGE16(
    uint gtid,
    uint partIndex,
    KeyStruct keys,
    OffsetStruct offsets)
{
#if defined(SORT_PAIRS)
    ScatterPairsDevicePartialWGE16(
        gtid,
        partIndex,
        keys,
        offsets);
#else
    ScatterKeysOnlyDevicePartial(gtid, partIndex);
#endif
}

inline void ScatterDevicePartialWLT16(
    uint gtid,
    uint partIndex,
    uint serialIterations,
    KeyStruct keys,
    OffsetStruct offsets)
{
#if defined(SORT_PAIRS)
    ScatterPairsDevicePartialWLT16(
        gtid,
        partIndex,
        serialIterations,
        keys,
        offsets);
#else
    ScatterKeysOnlyDevicePartial(gtid, partIndex);
#endif
}