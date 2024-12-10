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
var<storage, read_write> err: array<atomic<u32>>;

const BLOCK_DIM = 256u;
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    for(var i = id.x + 1u; i < info.size; i += griddim.x * BLOCK_DIM){
        if(sort[i - 1u] > sort[i]) {
            atomicAdd(&err[0], 1u);
        }
    }
}
