# GPUSorting

GPUSorting aims to bring state-of-the-art GPU sorting techniques from CUDA and make them available in portable compute shaders. All sorting algorithms included in GPUSorting utilize wave/warp/subgroup (referred to as "wave" hereon) level parallelism but are completely agnostic of wave size. Wave size specialization is entirely accomplished through runtime logic, instead of through shader compilation defines. This has a minimal impact on performance and significantly reduces the number of shader permutations. Although GPUSorting aims to be portable to any wave size supported by HLSL, [4, 128], due to hardware limitations, it has only been tested on wave sizes 4, 16, 32, and 64. You have been warned!

## Device Radix Sort vs OneSweep 

GPUSorting includes two sorting algorithms, both based on those found in the CUB library: DeviceRadixSort and OneSweep. The two algorithms are almost identical, except for the way that the inter-threadblock prefix sum of digit counts is performed. In DeviceRadixSort, the prefix sum is done through an older technique, "reduce-then-scan," whereas in OneSweep, it is accomplished using "[chained-scan-with-decoupled-lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)." Because "chained-scan" relies on forward thread-progress guarantees, OneSweep is less portable than DeviceRadixSort, and DeviceRadixSort should be used whenever portability is a concern. Again, due to a lack of hardware, I cannot say exactly how portable OneSweep is, but as a general rule of thumb, OneSweep appears to run correctly on anything that is not mobile or WARP. Use OneSweep at your own risk; you have been warned!

# Performance

As a measure of the quality of the code, GPUSorting has also been implemented in CUDA and benchmarked against Nvidia's [CUB](https://github.com/NVIDIA/cccl) library, with the following results:

![GPUSorting vs CUB](https://github.com/b0nes164/GPUSorting/assets/68340554/421ac4a3-6077-45af-b4cf-d27395b4d9b1)

## Thearling and Smith Benchmark:
![OneSweep Thearling and Smith Benchmark](https://github.com/b0nes164/GPUSorting/assets/68340554/3007376f-2cdc-4698-869f-d6b2f4a127dc)

## GPUSorting vs Fidelity FX Parallel Sort

In addition to CUB, GPUSorting has also been benchmarked against AMD's FidelityFX Parallel Sort (FFXPS), which is an algorithm much more familiar to game developers and has been included in the D3D12 implementation. FFXPS is also an LSD radix sort, but uses older techniques, in particular, a 4-bit radix. As opposed to GPUSorting, which uses 8-bit radixes, FFXPS must make 8 sorting passes to sort a 32-bit key. Another key difference is that FFXPS uses a fixed number of thread blocks per dispatch, which causes performance to suffer at larger input sizes (note the degradation in performance of pair sorting in particular). This is discussed in more detail in the section below titled "Handling Very Large Inputs." To ensure a fair comparison between algorithms, FFXPS was tuned to 256 threads per thread block and 1024 thread blocks. Tuning of FFXPS could probably be more aggressive; however, FFXPS is notably more brittle than GPUSorting when it comes to tuning, as loading of elements is not looped and some of the prefix sums appear to handle a maximum of 1024 elements. Increasing thread blocks beyond 1024 or increasing the partition size beyond 1024 resulted in crashes.

![GPUSorting vs FidelityFX Keys Only](https://github.com/b0nes164/GPUSorting/assets/68340554/195741cf-4f6d-42bb-bd03-93ddbe202373)

![GPUSorting vs FidelityFX Pairs](https://github.com/b0nes164/GPUSorting/assets/68340554/5aab9d57-5b47-407b-bd98-a9f68147f53f)

## Automatic Tuning for Devices:
GPUSorting is constructed in such a way that elements processed per thread, threads per thread block, and shared memory allocation can all be controlled through compiler defines. This effectively gives the runtime process control over all the necessary parameters required to tune the shader to a given device. Upon initializing an adapter, we identify the device and then check if we have a tuning preset for it. Currently, there are tuning presets for all non-workstation Nvidia cards Pascal+ and all non-workstation AMD cards RDNA1+. If no tuning preset is found for a device, a less aggressive generic preset is used.

Generally speaking, tuning presets are developed on a per-architecture basis. The general idea is to oversubscribe the SMs/WGPs as much as possible while maintaining as high occupancy as possible, maintaining an efficient pattern of memory accesses, and balancing the workload between waves in a thread block. On Nvidia cards, shared memory per SM sometimes varies within a given architecture, and so this is taken into account. For AMD cards, tuning is much more straightforward, as we only tune for RDNA1+, and the only difference between RDNA generations appears to be the reduction in wave hosting capacity on the SIMD32s from 20 to 16. For RDNA cards, we use the `WaveSize()` attribute to lock the wave size to 32, as we want WGPs not CUs.

![GPUSorting D3D12 Speeds, with Device-Based Tuning](https://github.com/b0nes164/GPUSorting/assets/68340554/e6c25fd8-23e8-47f4-99c2-5bb4e79b4eaf)

# Portability

### Handling Very Large Inputs

In D3D12, a compute shader dispatch is limited to a size of [65535 in all dimensions](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-dispatch). While Vulkan does support larger dispatches in the X dimension, this support is dependent on the capabilities of the device, and so for maximum portability, all dispatches should be treated as though 65535 is the limit in a single dimension. Given that we assign thread block partitions of sizes between $\left[ 3584, 7680 \right]$, we run into the problem that for very large dispatches, we are above the single dimension limit. For example, given a partition size of $3584$ and an input of size $2^{28}$, we would require $\left\lceil \frac{2^{28}}{3584} \right\rceil = 74899$, which is over the dimension limit. So how do we solve this?

In AMD's Fidelity FX, the approach taken is to choose a fixed amount of thread blocks and give each of them a variable amount of partitions to consume. However, this approach introduces a far more pernicious problem than it solves: It is now the responsibility of the implementer to either create a tuning mechanism to determine the appropriate number of thread blocks for a given device or to determine a "middle-of-the-road" thread block preset which is likely to be suboptimal on some devices.

Beyond the simple occupancy issues identified by the Fidelity FX authors, assigning multiple partitions to a single thread block introduces the more subtle problem that you are essentially forcing the GPU to consume partitions in a striped manner, potentially disrupting memory access patterns and cache efficiency. This problem is highlighted in the comparison above, where above input size $2^{26}$ FFX performance decreases dramatically despite correct execution. In fact, this slowdown is so significant that we can reliably reproduce TDR crashes given suboptimal-enough tuning parameters. And without a tuning mechanism to choose a sufficiently optimal preset, any input can potentially TDR crash a device.

For example, on my 2080 Super, I can reliably TDR by setting `maxThreadGroupsToRun` to 256 (which is still large enough to achieve full occupancy using 256 threads per thread block) and running input sizes of $2^{28}$. While this is a somewhat extreme example, on less capable hardware the TDR threshold is likely to be far lower.

GPUSorting solves the max dimension problem by expanding the dispatch into the Y dimension, and then making two dispatches instead of one: a dispatch of many maximum X dimensions, and a second partial dispatch of any remainders. A single root constant is used to indicate whether a thread block launch is in the "full" dispatch or in the "partial," and the appropriate block id is calculated by flattening the value. Although we are now potentially launching two dispatches, the performance penalty is extremely minimal (less than one percent), and for workloads below the dimension limit, only a single dispatch will be launched. An example of this can be found [here](https://github.com/b0nes164/GPUSorting/blob/bdf20a3f29ae47a195442f83cbbdedde583bde37/GPUSortingD3D12/DeviceRadixSortKernels.h#L79-L121).

### WaveMatch()

In HLSL Shader Model 6.5, Microsoft introduced the WaveMatch() intrinsic function. Basically, WaveMatch() compares the value of the expression and returns a bitmask of any other lanes in the wave that match this lane's value. However, given that we are only matching 8 bits at a time when ranking keys, and in order to maximize portability, we eschew WaveMatch() in favor of raw multisplitting.

# Getting Started

## GPUSortingD3D12

Headless implementation in D3D12, currently demo only, but release as a package is planned.

Requirements:
* Visual Studio 2019 or greater
* Windows SDK 10.0.20348.0 or greater

The repository folder contains a Visual Studio 2019 project and solution file. Upon building the solution, NuGet will download and link the external dependencies. See the repository wiki for information on running tests.

## GPUSortingCUDA

The purpose of this implementation is to benchmark the algorithms and demystify their implementation in the CUDA environment. It is not intended for production or use; instead, a proper implementation can be found in the CUB library.

* Visual Studio 2019 or greater
* Windows SDK 10.0.20348.0 or greater
* CUDA Toolkit 12.3.2
* Nvidia Graphics Card with Compute Capability 7.x or greater.

The repository folder contains a Visual Studio 2019 project and solution file; there are no external dependencies besides the CUDA toolkit. The use of sync primitives necessitates Compute Capability 7.x or greater. See the repository wiki for information on running tests.

## GPUSortingUnity

Released as a Unity package.

Requirements:
* Unity 2021.3.35f1 or greater

Within the Unity package manager, add a package from git URL and enter:

`https://github.com/b0nes164/GPUSorting.git?path=/GPUSortingUnity`

See the repository wiki for information on running tests.

# Strongly Suggested Reading / Bibliography

Andy Adinets and Duane Merrill. Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. 2022. arXiv: 2206.01784 url: https://arxiv.org/abs/2206.01784

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Saman Ashkiani et al. “GPU Multisplit”. In: SIGPLAN Not. 51.8 (Feb. 2016). issn: 0362-1340. doi: 10.1145/3016078.2851169. url: https://doi.org/10.1145/3016078.2851169.
