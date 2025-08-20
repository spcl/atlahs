# GOAL Generation for AI (NCCL) Applications

The GOAL generation toolchain for AI/NCCL applications was developed by [Zhiyi Hu](https://github.com/ZhiyiHu1999), and all credit goes to him. In addition, this tool was part of the project that focuses on analyzing the implementation of the protocols and collective algorithms implemented in NCCL, the details of which can be found in the paper [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786v1) (HOTI'25).

The original toolchain is also available at [nccl_goal_generator](https://github.com/ZhiyiHu1999/nccl_goal_generator).

## Introduction

**nccl_goal_generator** is a toolchain for **tracing and parsing [NCCL](https://github.com/NVIDIA/nccl) communication** data, producing a compiled `GOAL` file.

This toolchain includes:

1. **NCCL Tracer** – intercepts and records NCCL calls and events (via [nccl_nvtx_v2.20.5-1](https://github.com/ZhiyiHu1999/nccl_nvtx_v2.20.5-1) and [Nsight Systems](https://developer.nvidia.com/nsight-systems)).
2. **Trace Parser** – converts the collected NCCL trace into a `GOAL` file compatible with [LogGOPSim](https://github.com/spcl/LogGOPSim).
3. **Sample scripts and examples** – show how to enable tracing in GPU applications, as well as how to process and visualize the resulting data.

> **Disclaimer**: This project is **not** affiliated with or endorsed by [NVIDIA](https://www.nvidia.com/).
> **NCCL** and **NVTX** remain the property of NVIDIA or its affiliates.
> This toolchain simply extends and integrates these libraries to enable custom tracing and simulation workflows.

---

## Features & Workflow

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

## Dependencies & Installation

### Install Required Dependencies

1. **nccl_nvtx_v2.20.5-1** (NCCL with NVTX annotations)

   - The toolchain uses a custom version of NCCL with additional NVTX annotations.
   - To install, please use the script `make_ault.sh` and `make_daint.sh` in the `nccl_nvtx_v2.20.5` directory as a reference. When you are compiling the NCCL library, make sure to configure `CUDA_HOME` to the correct path on your system.

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

## Usage Introduction

### Usage Steps

To use the toolchain, firstly, make sure that your application runs correctly without the toolchain tracing. Then, you can modify the command that runs the training script to use the module in the toolchain.

  1. Set `LD_PRELOAD` to the nvtx-annotated nccl given in the toolchain. (Require minor modifications based on your path)

  ```bash
  export LD_PRELOAD=$HOME/nccl_goal_generator/nccl_nvtx_v2.20.5/nccl/build/lib/libnccl.so
  ```

  2. Add `nsys profile` to the command that runs the training script. The recommended arguments for the `nsys profile` command are:
   
   ```bash
   nsys profile --trace='nvtx,cuda' \
                --cuda-memory-usage=false \
                --force-overwrite true \
                --cuda-um-cpu-page-faults=false \
                --cuda-um-gpu-page-faults=false \
                -s none \
                --output='${TRACE_DIR}/nsys_report_%h_%p.nsys-rep'
   ```
   where `TRACE_DIR` is the directory where the nsys report will be saved. You can use the `run_megatron.sh` script in the `apps/ai/scripts` directory as a reference.

  3. Run the modified command.

  4. Once the job completes, the nsys report will be saved in the directory specified by the `output` argument.

  5. Make sure to convert the nsys reports to sqlite files first with the `nsys_reports_to_sqlite.sh` script in the `apps/ai/scripts` directory.
  6. Use `nccl_goal_generator/get_traced_events.py` to convert the `sqlite` files to a `GOAL` file. The arguments of the script are:

   | Argument                   | Short | Type   | Required | Default | Description                                                         |
   | -------------------------- | ----- | ------ | -------- | ------- | ------------------------------------------------------------------- |
   | `--trace-dir`              | `-i`  | string | Yes      | -       | Directory containing the nsys profiles of the application           |
   | `--output-dir`             | `-o`  | string | Yes      | -       | Output directory to store the goal files and intermediate files     |
   | `--no-intermediate-output` | `-q`  | flag   | No       | False   | Do not generate intermediate output files                           |
   | `--config_node_gpu`        | `-c`  | string | No       | -       | YAML file for configuration of nodes and GPUs                       |
   | `--npkit_file_Simple`      | `-s`  | string | No       | -       | NPKit benchmark results JSON file for Simple Protocol               |
   | `--npkit_file_LL`          | `-l`  | string | No       | -       | NPKit benchmark results JSON file for LL Protocol                   |
   | `--zero-red-copy`          | -     | flag   | No       | False   | Whether to set all the reduction copy time to zero                  |
   | `--merge-non-overlap`      | -     | flag   | No       | False   | Whether to merge non-overlapping events for all streams if possible |
   | `--unique-nic`             | -     | flag   | No       | False   | Whether to assign a separate NIC ID for each GPU in GOAL            |

   Set the `trace-dir` to the directory where the sqlite files are saved. The `output-dir` is the directory where the `GOAL` file will be saved. After the script finishes successfully, you will find the GOAL file as `InterNode_MicroEvents_Dependency.gol`.


## Profiling

One thing to note is that in order to generate GOAL files whose compute vertices accurately represent the actual computation in specific GPUs, we need to profile the time it takes for the reduction and copy operations. That is the reason why we provide the `--npkit_file_Simple` and `--npkit_file_LL` arguments. The `npkit_file_Simple` and `npkit_file_LL` are the JSON files that contain the profiling results of the reduction and copy operations for the Simple and LL protocols, respectively.

We provide the results of the profiling in the `npkit_benchmark_results` directory for two clusters maintained by the Swiss National Supercomputing Center (CSCS):
- **Testbed cluster Ault**: RTX 3090
- **Alps cluster Daint**: GH200 Superchip (NVIDIA H100)

To collect the profiling results on your own system, you can use the scripts in `npkit_benchmark` as a reference.








