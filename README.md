# GPUSorting

GPUSorting aims to bring state of the art GPU sorting techniques from CUDA and make them available in portable compute shaders. All sorting algorithms included in GPUSorting utilize wave/warp/subgroup (will be referred to as "wave" from here on) level parallelism, but are completely agnostic of wave size. Instead of accomplishing this through shader compilation variants, this done entirely with runtime logic, with minimal performance penalties. Although GPUSorting aims to be portable to any wave size supported by HLSL, [4, 128], due to hardware limitations it has only been tested on wave sizes 4, 16, 32, and 64. You have been warned!

### Portability
GPUSorting includes two sorting algorithms, both based off of those found in the CUB library: DeviceRadixSort and OneSweep. The two algorithms are almost identical, except for the way that the inter-threadblock prefix sum of digit counts is performed. In DeviceRadixSort the prefix sum is done through an older technique, "reduce-then-scan," whereas in OneSweep it is accomplished using "chained-scan-with-decoupled-lookback." Because "chained-scan" relies on forward thread-progress requirements, OneSweep is less portable than DeviceRadixSort, and DeviceRadixSort should be used whenever portability is a concern. Again, due to a lack of hardware, I cannot say exactly how portable OneSweep is, but as a general rule of thumb, OneSweep ***appears*** to run correctly on anything that is not mobile or WARP. Use OneSweep at your own risk, you have been warned! 

### Performance
As a measure of the quality of the code, GPUSorting has also been implemented in CUDA, and benchmarked againts Nvidia's [CUB](https://github.com/NVIDIA/cccl) library, with the following results:

![GPUSorting vs CUB](https://github.com/b0nes164/GPUSorting/assets/68340554/ab29deec-25bb-438c-81f6-3c121b1e1638)

Currently, GPUSorting does not incorporate any sort of device-based tuning, instead we use a "middle-of-the-road" tuning preset that appears to work well on most devices:

![GPUSorting D3D12 Speeds](https://github.com/b0nes164/GPUSorting/assets/68340554/df6fe396-339f-4061-8d55-94f4cf61dd48)

# Implementations Overview

## GPUSortingD3D12

Headless implementation in D3D12, currently demo only, but release as a package planned.

## GPUSortingCUDA

The purpose of this implementation is to benchmark the algorithms and demystify their implementation in the CUDA environment. It is not intended for production or use, instead a proper implementation can be found in the CUB library.

## GPUSortingUnity

Released as a Unity package.

# Getting Started

## GPUSortingD3D12

Requirements:
* Visual Studio 2019 or greater
* Windows SDK 10.0.20348.0 or greater

The repository folder contains a Visual Studio 2019 project and solution file. Upon building the solution, NuGet will download and link the external depedencies.
See the repository wiki for information on running tests.

## GPUSortingCUDA
* Visual Studio 2019 or greater
* Windows SDK 10.0.20348.0 or greater
* CUDA Toolkit 12.3.2
* Nvidia Graphics Card with Compute Capability 7.x or greater.
  
The repository folder contains a Visual Studio 2019 project and solution file, there are no external depedencies besides the CUDA toolkit. Use of sync primitives necessitates Compute Capability 7.x or greater. See the repository wiki for information on running tests.

## GPUSortingUnity

Requirements: 
* Unity 2021.3.35f1 or greater

Within the Unity package manager add a package from git url and enter:

`https://github.com/b0nes164/GPUSorting.git?path=/GPUSortingUnity`

See the repository wiki for information on running tests.

# Strongly Suggested Reading / Bibliography

Andy Adinets and Duane Merrill. Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. 2022. arXiv: 2206.01784 url: https://arxiv.org/abs/2206.01784

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Saman Ashkiani et al. “GPU Multisplit”. In: SIGPLAN Not. 51.8 (Feb. 2016). issn: 0362-1340. doi: 10.1145/3016078.2851169. url: https://doi.org/10.1145/3016078.2851169.
