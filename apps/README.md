Applications
==================

This directory contains a collection of applications that we have used to test the toolchain. There are currently two types of applications, namely AI and HPC,which are detailed below. As for storage, we have not yet included applications that target storage specifically to test the toolchain, and primarily relied on the [UMass storage traces](https://traces.cs.umass.edu/docs/traces/storage/). Note that we are in the process of including more applications, and will update this README as we go.

1. AI applications (`ai/`):
    - [DLRM](https://github.com/facebookresearch/dlrm): A deep learning recommendation model.
    - [Megatron](https://github.com/NVIDIA/Megatron): Workloads for LLM training.
    - [MMaDA](https://github.com/NVIDIA/Megatron): Workloads for multi-model diffusion LLM.
    - [vLLM](https://github.com/vllm-project/vllm): Workloads for LLM inference.
2. HPC applications (`hpc/`):
    - [HPCG](https://github.com/hpcg-benchmark/hpcg): High-Performance Conjugate Gradient (HPCG) benchmark.
    - [LULESH](https://github.com/LLNL/LULESH): Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics (LULESH) benchmark.
    - [ICON](https://gitlab.dkrz.de/icon/icon-model): Workloads for climate model.
    - [LAMMPS](https://github.com/lammps/lammps): Workloads for molecular dynamics simulation.
    - [MILC_QCD](https://github.com/milc-qcd/milc_qcd): Workloads for lattice quantum chromodynamics (QCD) simulation.
    - [NPB](https://www.nas.nasa.gov/software/npb.html): NAS Parallel Benchmarks.
    - [OpenMX](https://github.com/OpenMX-org/openmx): Workloads for nano-scale material simulation.
    - [CloverLeaf](https://uk-mac.github.io/CloverLeaf/): A Lagrangian-Eulerian hydrodynamics benchmark
