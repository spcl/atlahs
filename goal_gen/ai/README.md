# nccl_goal_generator

## 1. Introduction

**nccl_goal_generator** is a toolchain for **tracing and parsing [NCCL](https://github.com/NVIDIA/nccl) communication** data, producing a compiled `GOAL` file that can be fed into [LogGOPSim](https://github.com/spcl/LogGOPSim) for large-scale network simulation.

This toolchain includes:

1. **NCCL Tracer** – intercepts and records NCCL calls and events (via [nccl_nvtx_v2.20.5-1](https://github.com/ZhiyiHu1999/nccl_nvtx_v2.20.5-1) and [Nsight Systems](https://developer.nvidia.com/nsight-systems)).
2. **Trace Parser** – converts the collected NCCL trace into a `GOAL` file compatible with [LogGOPSim](https://github.com/spcl/LogGOPSim).
3. **Sample scripts and examples** – show how to enable tracing in GPU applications, as well as how to process and visualize the resulting data.

> **Disclaimer**: This project is **not** affiliated with or endorsed by [NVIDIA](https://www.nvidia.com/).
> **NCCL** and **NVTX** remain the property of NVIDIA or its affiliates.
> This toolchain simply extends and integrates these libraries to enable custom tracing and simulation workflows.

---

## 2. Features & Workflow

1. **Tracing NCCL Communication**

   - During distributed GPU execution, insert tracing hooks (e.g., via [nccl_nvtx](https://github.com/ZhiyiHu1999/nccl_nvtx_v2.20.5-1)) to record all major NCCL calls along with timestamps.
   - Tracing pairs with [Nsight Systems](https://developer.nvidia.com/nsight-systems) to gather NCCL invocation data.
2. **Trace Parsing & Goal File Generation**

   - Once the GPU job completes, nsys reports including NCCL traces are produced.
   - Our parser reads this trace and outputs a LogGOPSim-compatible `GOAL` file.
3. **Offline Network Simulation with LogGOPSim**

   - Pass the resulting `GOAL` file to [LogGOPSim](https://github.com/spcl/LogGOPSim) to simulate large-scale network topologies and evaluate performance bottlenecks, scalability, bandwidth usage, and more.
   - Compare simulation outputs to real-world measurements to refine your network design or distribution strategy.

---

## 3. Dependencies & Installation

### 3.1 Cloning this Repository

To clone this repository along with its submodules, use:

```bash
git clone --recursive https://github.com/ZhiyiHu1999/nccl_goal_generator
cd nccl_goal_generator
```

### 3.2 Install Required Dependencies

1. **nccl_nvtx_v2.20.5-1** (NCCL with  NVTX annotations)

   - [Repository](https://github.com/ZhiyiHu1999/nccl_nvtx_v2.20.5-1)
   - Used as a submodule.
   - To install:

   ```bash
   cd third_party/nccl_nvtx
   bash make_ault.sh
   ```
2. **Nsight Systems**

   - For profiling, post-processing, and parsing nccl traces, the tracer should work with [Nsight Systems](https://developer.nvidia.com/nsight-systems)
   - Recommanded Version: `Nsight Systems 2024.5.1`
   - To install:

   ```bash
   wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/NsightSystems-linux-cli-public-2024.5.1.113-3461954.rpm
   rpm2cpio NsightSystems-linux-cli-public-2024.5.1.113-3461954.rpm | cpio -idmv

   ## Necessary and may needs minor modifications based on your nsys version:
   echo 'export PATH=~/opt/nvidia/nsight-systems-cli/2024.5.1/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc

   ## To verify the installation:
   nsys --version
   ## A successful installation should output like:
   NVIDIA Nsight Systems version 2020.4.3.7-10543b6

   ## To verify the installation:
   which nsys
   ## A successful installation should output like:
   ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys
   ```
3. **LogGOPSim**

   - [Repository](https://github.com/spcl/LogGOPSim)
   - Used as a submodule.
   - Consumes the `goal` file generated for network simulation.
   - Necessary setting (may require minor modifications based on your path):

   ```bash
   echo 'export PATH=$HOME/nccl_goal_generator/third_party/LogGOPSim-1.1:$PATH' >> ~/.bashrc
   source ~/.bashrc

   ## To verify the installation:
   which txt2bin
   ## A successful installation should output like:
   ~/nccl_goal_generator/third_party/LogGOPSim-1.0/txt2bin

   ## To verify the installation:
   which LogGOPSim 
   ## A successful installation should output like:
   ~/nccl_goal_generator/third_party/LogGOPSim-1.0/LogGOPSim
   ```

### 3.3 Install nccl_goal_generator

```bash
   cd $HOME/nccl_goal_generator
   pip install -e .

   ## To verify the installation:
   which nccl_goal_generator 
   ## A successful installation should output like:
   ~/anaconda3/bin/nccl_goal_generator
```

## 4. Usage Introduction

### 4.1 Usage Steps

Steps to use the toolchain include:

- Step 1: Make sure your job script can run successfully without the toolchain.
- Step 2: Modify in the job script after step 1.

  1. Set `LD_PRELOAD` to the nvtx-annotated nccl given in the toolchain. (Require minor modifications based on your path)

  ```bash
  export LD_PRELOAD=$HOME/nccl_goal_generator/third_party/nccl_nvtx/nccl/build/lib/libnccl.so
  ```

  2. Modify the command that runs the training script to use the module in the toolchain

  ```bash
  ## Original command to run the script
  srun bash run_ds.sh

  ## Modified command that uses the toolchain
  nccl_goal_generator --training_script run_ds.sh --results_dir results --config_node_gpu node_gpu_config.yaml
  ```

### 4.2 Arguments

- `--training_script`: Path to the training script that can run in step 1.
- `--results_dir`: Path for the compiled goal file (bin file).
- `--config_node_gpu`: Path for the user specified nodes and GPUs configuration. (The total number of GPUs you specify should be equal to the GPUs used for tracing)

## 5. TODO

- ~~Apply NPKit estimated probablistic model for reduction time to trace generator~~
- ~~For different streams on a node, add `cpu <streamID>` for the calc time~~
