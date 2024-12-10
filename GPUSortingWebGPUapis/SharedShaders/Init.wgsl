//****************************************************************************
// GPUSorting
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 12/7/2024
// https://github.com/b0nes164/GPUSorting
//
//****************************************************************************

struct InfoStruct
{
    size: u32,
    shift: u32,
    thread_blocks: u32,
    seed: u32,
};

@group(0) @binding(0)
var<uniform> info : InfoStruct; 

@group(0) @binding(1)
var<storage, read_write> bump: array<u32>;

@group(0) @binding(2)
var<storage, read_write> sort: array<u32>;

@group(0) @binding(3)
var<storage, read_write> alt: array<u32>;

@group(0) @binding(4)
var<storage, read_write> payload: array<u32>;

@group(0) @binding(5)
var<storage, read_write> alt_payload: array<u32>;

@group(0) @binding(6)
var<storage, read_write> hist: array<u32>;

@group(0) @binding(7)
var<storage, read_write> pass_hist: array<u32>;

@group(0) @binding(8)
var<storage, read_write> err: array<u32>;

fn update_rng_state(state: vec4<u32>) -> vec4<u32> {
    var res: vec4<u32>;
    res.x = ((state.x & 4294967294u) << 12u) ^ (((state.x << 13u) ^ state.x) >> 19u);
    res.y = ((state.y & 4294967288u) << 4u) ^ (((state.y << 2u) ^ state.y) >> 25u);
    res.z = ((state.z & 4294967280u) << 17u) ^ (((state.z << 3u) ^ state.z) >> 11u);
    res.w = state.w * 1664525u + 1013904223u;
    return res;
}

fn get_rng(state: vec4<u32>) -> u32 {
    return state.x ^ state.y ^ state.z ^ state.w;
}

const BLOCK_DIM = 256u;
const RADIX = 256u;
const SORT_PASSES = 4u;
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    var state = vec4<u32>(
        (id.x * 4) * info.seed,
        (id.x * 4 + 1) * info.seed,
        (id.x * 4 + 2) * info.seed,
        (id.x * 4 + 3) * info.seed);

    for(var i = id.x; i < info.size; i += griddim.x * BLOCK_DIM){
        state = update_rng_state(state);
        let t = get_rng(state);
        sort[i] = t;
        payload[i] = t;
    }

    for(var i = id.x; i < info.thread_blocks * RADIX * SORT_PASSES; i += griddim.x * BLOCK_DIM){
        pass_hist[i] = 0u;
    }

    for(var i = id.x; i < RADIX * SORT_PASSES; i += griddim.x * BLOCK_DIM) {
        hist[i] = 0u;
    }

    if(id.x <= SORT_PASSES){
        bump[id.x] = 0u;
    }
}
