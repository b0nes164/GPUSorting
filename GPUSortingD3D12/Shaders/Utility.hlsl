/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/13/2024
 * https://github.com/b0nes164/GPUSorting
 * 
 ******************************************************************************/

#define VAL_PART_SIZE   2048
#define VAL_THREADS     256

#if defined(KEY_UINT)
RWStructuredBuffer<uint> b_sort         : register(u0);
#elif defined(KEY_INT)
RWStructuredBuffer<int> b_sort          : register(u0);
#elif defined(KEY_FLOAT)
RWStructuredBuffer<float> b_sort        : register(u0);
#endif

#if defined(PAYLOAD_UINT)
RWStructuredBuffer<uint> b_sortPayload  : register(u1);
#elif defined(PAYLOAD_INT)
RWStructuredBuffer<int> b_sortPayload   : register(u1);
#elif defined(PAYLOAD_FLOAT)
RWStructuredBuffer<float> b_sortPayload : register(u1);
#endif

RWStructuredBuffer<uint> b_passHist : register(u2);
RWStructuredBuffer<uint> b_errorCount : register(u3);

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_threadBlocks;
    uint e_seed;
    uint e_andCount;
};

#if defined(KEY_UINT)
groupshared uint g_val[VAL_PART_SIZE + 1];
#elif defined(KEY_INT)
groupshared int g_val[VAL_PART_SIZE + 1];
#elif defined(KEY_FLOAT)
groupshared float g_val[VAL_PART_SIZE + 1];
#endif

#if defined(PAYLOAD_UINT)
groupshared uint g_valPayload[VAL_PART_SIZE + 1];
#elif defined(PAYLOAD_INT)
groupshared int g_valPayload[VAL_PART_SIZE + 1];
#elif defined(PAYLOAD_FLOAT)
groupshared float g_valPayload[VAL_PART_SIZE + 1];
#endif

//Hybrid Tausworthe
//GPU GEMS CH37 Lee Howes + David Thomas
#define TAUS_STEP_1 ((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19)
#define TAUS_STEP_2 ((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25)
#define TAUS_STEP_3 ((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11)
#define LCG_STEP    (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS (z1 ^ z2 ^ z3 ^ z4)

//Initialize the input on the GPU, assumes threadblocks = 256
//Match the payload to the output (on a bit-level)
//That way, if the payload was sorting correctly, we can check
//In the same way we check the keys.
//This is a bit of a hack (does not directly test stability of the payloads), 
//but if sort was not stable, the keys would fail, and payloads would fail

[numthreads(VAL_THREADS, 1, 1)]
void InitSortInput(int3 id : SV_DispatchThreadID)
{
    const uint numKeys = e_numKeys;
    const uint inc = VAL_THREADS * 256;
    
    uint z1 = (id.x << 2) * e_seed;
    uint z2 = ((id.x << 2) + 1) * e_seed;
    uint z3 = ((id.x << 2) + 2) * e_seed;
    uint z4 = ((id.x << 2) + 3) * e_seed;
    
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;
    
    for (uint i = id.x; i < numKeys; i += inc)
    {
        z1 = TAUS_STEP_1;
        z2 = TAUS_STEP_2;
        z3 = TAUS_STEP_3;
        z4 = LCG_STEP;
#if defined(KEY_UINT)
        b_sort[i] = HYBRID_TAUS;
#elif defined (KEY_INT)
        b_sort[i] = asint(HYBRID_TAUS);
#elif defined (KEY_FLOAT)
        b_sort[i] = asfloat(HYBRID_TAUS);
#endif
        
#if defined(SORT_PAIRS)
    #if defined(PAYLOAD_UINT)
        b_sortPayload[i] = HYBRID_TAUS;
    #elif defined (PAYLOAD_INT)
        b_sortPayload[i] = asint(HYBRID_TAUS);
    #elif defined (PAYLOAD_FLOAT)
        b_sortPayload[i] = asfloat(HYBRID_TAUS);
    #endif
#endif
    }
}

//An Improved Supercomputer Sorting Benchmark
//Kurt Thearling & Stephen Smith
//Bitwise AND successive keys together to decrease entropy
//in a way that is evenly distributed across histogramming
//passes.
//NOTE this will break payload validation, which assumes keys and
//payloads have identical values.
//Number of Keys ANDed | Entropy
//        1            |  1.0 bits
//        2            | .811 bits
//        3            | .544 bits
//        4            | .337 bits
//        5            | .201 bits
[numthreads(VAL_THREADS, 1, 1)]
void InitEntropyControlled(int3 id : SV_DispatchThreadID)
{
    const uint increment = VAL_THREADS * 256;
    const uint alignedEnd = e_numKeys / e_andCount;
    for (uint i = id.x * e_andCount; i < alignedEnd; i += increment)
    {
        for (uint k = 1; k < e_andCount; ++k)
        {
#if defined(KEY_UINT)
            b_sort[i + k] &= b_sort[i + k - 1];
#elif defined (KEY_INT) || defined (KEY_FLOAT)
            uint t = asuint(b_sort[i + k]);
            t &= asuint(b_sort[i + k - 1]);
#endif
#if defined(KEY_INT)
            b_sort[i + k] = asint(t);
#elif defined(KEY_FLOAT)
            b_sort[i + k] = asfloat(t);
#endif
            
        } 
    }

    if (!id.x)
    {
        for (uint k = 1; k < e_andCount; ++k)
        {
            if (alignedEnd + k < e_numKeys)
            {
#if defined(KEY_UINT)
            b_sort[alignedEnd + k] &= b_sort[alignedEnd + k - 1];
#elif defined (KEY_INT) || defined (KEY_FLOAT)
            uint t = asuint(b_sort[alignedEnd + k]);
            t &= asuint(b_sort[alignedEnd + k - 1]);
#endif
#if defined(KEY_INT)
            b_sort[alignedEnd + k] = asint(t);
#elif defined(KEY_FLOAT)
            b_sort[alignedEnd + k] = asfloat(t);
#endif
            }
        }
    }
}

//Used to standalone test the scan kernel, assumes threadblocks = 1
//Scan values so small its not a huge time cost to check on the CPU
[numthreads(VAL_THREADS, 1, 1)]
void InitScanTestValues(int3 id : SV_DispatchThreadID)
{
    if (id.x < e_numKeys)
        b_passHist[id.x] = 1;
}

[numthreads(1, 1, 1)]
void ClearErrorCount(int3 id : SV_DispatchThreadID)
{
    b_errorCount[0] = 0;
}

[numthreads(VAL_THREADS, 1, 1)]
void Validate(int3 gtid : SV_GroupThreadID, int3 gid : SV_GroupID)
{
    if (gid.x < e_threadBlocks - 1)
    {
        const uint t = gid.x * VAL_PART_SIZE;
        for (int i = gtid.x; i < VAL_PART_SIZE + 1; i += VAL_THREADS)
        {
#if defined(KEY_UINT) || defined (KEY_INT) || defined (KEY_FLOAT)
            g_val[i] = b_sort[i + t];
#endif
#if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined (PAYLOAD_INT) || defined (PAYLOAD_FLOAT))
            g_valPayload[i] = b_sortPayload[i + t];
#endif
        }
        GroupMemoryBarrierWithGroupSync();
        
        //Reinterpret the payload to match the type of the key it was sorted on
        for (int i = gtid.x; i < VAL_PART_SIZE; i += VAL_THREADS)
        {
            uint isInvalid = 0;
#if defined(KEY_UINT) || defined(KEY_INT) || defined(KEY_FLOAT)
    #if defined(SHOULD_ASCEND)
            isInvalid |= g_val[i] > g_val[i + 1];
        #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
            #if defined (KEY_UINT)
            isInvalid |= asuint(g_valPayload[i]) > asuint(g_valPayload[i + 1]);
            #elif defined(KEY_INT)
            isInvalid |= asint(g_valPayload[i]) > asint(g_valPayload[i + 1]);
            #elif defined(KEY_FLOAT)
            isInvalid |= asfloat(g_valPayload[i]) > asfloat(g_valPayload[i + 1]);
            #endif
        #endif
    #else
            isInvalid |= g_val[i] < g_val[i + 1];
        #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
            #if defined (KEY_UINT)
            isInvalid |= asuint(g_valPayload[i]) < asuint(g_valPayload[i + 1]);
            #elif defined(KEY_INT)
            isInvalid |= asint(g_valPayload[i]) < asint(g_valPayload[i + 1]);
            #elif defined(KEY_FLOAT)
            isInvalid |= asfloat(g_valPayload[i]) < asfloat(g_valPayload[i + 1]);
            #endif
        #endif
    #endif
#endif
            if (isInvalid)
                InterlockedAdd(b_errorCount[0], 1);
        }
    }
    else
    {
        for (int i = gtid.x + gid.x * VAL_PART_SIZE; i < e_numKeys - 1; i += VAL_THREADS)
        {
            uint isInvalid = 0;     
#if defined(KEY_UINT) || defined(KEY_INT) || defined(KEY_FLOAT)
    #if defined(SHOULD_ASCEND)
            isInvalid |= b_sort[i] > b_sort[i + 1];
        #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
            #if defined (KEY_UINT)
            isInvalid |= asuint(b_sortPayload[i]) > asuint(b_sortPayload[i + 1]);
            #elif defined(KEY_INT)
            isInvalid |= asint(b_sortPayload[i]) > asint(b_sortPayload[i + 1]);
            #elif defined(KEY_FLOAT)
            isInvalid |= asfloat(b_sortPayload[i]) > asfloat(b_sortPayload[i + 1]);
            #endif
        #endif
    #else
            isInvalid |= b_sort[i] < b_sort[i + 1];
        #if defined(SORT_PAIRS) && (defined(PAYLOAD_UINT) || defined(PAYLOAD_INT) || defined(PAYLOAD_FLOAT))
            #if defined (KEY_UINT)
            isInvalid |= asuint(b_sortPayload[i]) < asuint(b_sortPayload[i + 1]);
            #elif defined(KEY_INT)
            isInvalid |= asint(b_sortPayload[i]) < asint(b_sortPayload[i + 1]);
            #elif defined(KEY_FLOAT)
            isInvalid |= asfloat(b_sortPayload[i]) < asfloat(b_sortPayload[i + 1]);
            #endif
        #endif
    #endif
#endif
            if (isInvalid)
                InterlockedAdd(b_errorCount[0], 1);
        }
    }
}