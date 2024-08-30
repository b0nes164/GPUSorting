# GPUSorting

GPUSorting aims to bring state-of-the-art GPU sorting techniques from CUDA and make them available in portable compute shaders. All sorting algorithms included in GPUSorting utilize wave/warp/subgroup (referred to as "wave" hereon) level parallelism but are completely agnostic of wave size. Wave size specialization is entirely accomplished through runtime logic, instead of through shader compilation defines. This has a minimal impact on performance and significantly reduces the number of shader permutations. Although GPUSorting aims to be portable to any wave size supported by HLSL, [4, 128], due to hardware limitations, it has only been tested on wave sizes 4, 16, 32, and 64. You have been warned!

## Device Radix Sort vs OneSweep 

GPUSorting includes two sorting algorithms, both based on those found in the CUB library: _DeviceRadixSort_ and _OneSweep_. The two algorithms are almost identical, except for the way that the inter-threadblock prefix sum of digit counts is performed. In _DeviceRadixSort_, the prefix sum is done through an older technique, "reduce-then-scan," whereas in _OneSweep_, it is accomplished using "[chained-scan-with-decoupled-lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)." Because "chained-scan" relies on forward thread-progress guarantees, _OneSweep_ is less portable than _DeviceRadixSort_, and _DeviceRadixSort_ should be used whenever portability is a concern. Again, due to a lack of hardware, I cannot say exactly how portable _OneSweep_ is, but as a general rule of thumb, _OneSweep_ tends to run on anything that is not mobile, a software rasterizer, or Apple. Use _OneSweep_ at your own risk; you have been warned!

As a measure of the quality of the code, GPUSorting has also been implemented in CUDA and benchmarked against Nvidia's [CUB](https://github.com/NVIDIA/cccl) library, with the following results:

![GPUSorting vs CUB](https://github.com/b0nes164/GPUSorting/assets/68340554/421ac4a3-6077-45af-b4cf-d27395b4d9b1)

## SplitSort

GPUSorting also introduces a novel hybrid radix-merge based segmented sort called ___SplitSort___. Due to its unique radix-based property across all maximum segment lengths, _SplitSort_ demonstrates significant speedups when sorting on 16-bit keys. On 32-bit keys, _SplitSort_ shows modest speedups on maximum segment lengths less than 256, particularly when sorting with a 64-bit value. At this point, _SplitSort_ is still very much a proof of concept. For a more complete write-up on how _SplitSort_ works under the hood, see this thread on the [Lindbender Org Zulip](https://xi.zulipchat.com/#narrow/stream/197075-gpu/topic/A.20Better.20Sort.20for.20Sparse.20Strip.20Rendering). Note that the following benchmarks were performed on Ubuntu using the benchmarking suite provided by [Kobus et al.](https://gitlab.rlp.net/pararch/faster-segmented-sort-on-gpus).

![Segmented Sort Comparison, (uint32_t, uint32_t)](https://github.com/user-attachments/assets/5ef32d05-0f91-4bc1-a2f1-d4defb8b35fe)



![Segmented Sort Comparison, (uint32_t, double)(1)](https://github.com/user-attachments/assets/2853b27e-bdc9-4162-98f5-2607562d9a02)

# Various Other Benchmarks

## Thearling and Smith Benchmark:

![OneSweep Thearling and Smith Benchmark](https://github.com/b0nes164/GPUSorting/assets/68340554/3007376f-2cdc-4698-869f-d6b2f4a127dc)

## GPUSorting vs Fidelity FX Parallel Sort

![GPUSorting vs FidelityFX Keys Only](https://github.com/b0nes164/GPUSorting/assets/68340554/195741cf-4f6d-42bb-bd03-93ddbe202373)

![GPUSorting vs FidelityFX Pairs](https://github.com/b0nes164/GPUSorting/assets/68340554/5aab9d57-5b47-407b-bd98-a9f68147f53f)

## Automatic Tuning for Devices:

![GPUSorting D3D12 Speeds, with Device-Based Tuning](https://github.com/b0nes164/GPUSorting/assets/68340554/e6c25fd8-23e8-47f4-99c2-5bb4e79b4eaf)

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

Dondragmer. CuteSort. https://gist.github.com/dondragmer/0c0b3eed0f7c30f7391deb11121a5aa1.

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Kaixi Hou, Weifeng Liu, Hao Wang, and Wu-chun Feng. 2017. Fast segmented sort on GPUs. In Proceedings of the International Conference on Supercomputing (ICS '17). Association for Computing Machinery, New York, NY, USA, Article 12, 1–10. https://doi.org/10.1145/3079079.3079105

Kobus, R., Nelgen, J., Henkys, V., Schmidt, B. (2023). Faster Segmented Sort on GPUs. In: Cano, J., Dikaiakos, M.D., Papadopoulos, G.A., Pericàs, M., Sakellariou, R. (eds) Euro-Par 2023: Parallel Processing. Euro-Par 2023. Lecture Notes in Computer Science, vol 14100. Springer, Cham. https://doi.org/10.1007/978-3-031-39698-4_45

Oded Green, Robert McColl, and David A. Bader. 2012. GPU merge path: a GPU merging algorithm. In Proceedings of the 26th ACM international conference on Supercomputing (ICS '12). Association for Computing Machinery, New York, NY, USA, 331–340. https://doi.org/10.1145/2304576.2304621

Rafael F. Schmid, Flávia Pisani, Edson N. Cáceres, and Edson Borin. 2022. An evaluation of fast segmented sorting implementations on GPUs. Parallel Comput. 110, C (May 2022). https://doi.org/10.1016/j.parco.2021.102889

Saman Ashkiani et al. “GPU Multisplit”. In: SIGPLAN Not. 51.8 (Feb. 2016). issn: 0362-1340. doi: 10.1145/3016078.2851169. url: https://doi.org/10.1145/3016078.2851169.

Sean Baxter. Segmented Sort and Locality Sort. https://moderngpu.github.io/segsort.html
