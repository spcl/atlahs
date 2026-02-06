# Adding ATLAHS NVTX Annotations to NCCL

## Goal

Build NCCL with **ATLAHS NVTX annotations** so the ATLAHS pipeline (nsys → sqlite → GOAL → LGS) can reliably pick up NCCL activity.

This folder is where we keep NCCL-version-specific bits (patches or annotated source trees).

## Supported NCCL versions (current)

| NCCL version | How it is distributed here | Artifact |
|---|---|---|
| 2.20.x | Annotated source tree (historical baseline used by ATLAHS; distributed as modified source) | `nccl_nvtx_v2.20.5/` (git submodule) |
| 2.28.3-1 | Patch to apply to clean NCCL sources | `nccl_atlahs_228.patch` |

## Forward compatibility

Depending on upstream NCCL changes, a patch for one release may apply cleanly to a newer release (especially minor/patch updates), but this is not guaranteed.

Quick check:

- Try `git apply --check` against your target NCCL version.
- If it applies, rebuild and verify NVTX markers show up in the trace.
- If it does not apply, expect to update the patch around the enqueue/init/collectives codepaths.

## What the ATLAHS annotations add (NCCL behavior unchanged)

- NVTX ranges around:
   - collectives API
   - enqueue path (WorkElemColl / CollInfo metadata like `nWarps`, chunking, buffers)
   - group start/end
   - init/topology
- Build flags (passed via `TRACING_FLAGS`): `-DENABLE_API_NVTX -DENABLE_INIT_NVTX -DENABLE_ENQUEUE_NVTX`
- Typical files touched (paths vary by version):
   - `makefiles/common.mk`
   - `src/collectives.cc`, `src/enqueue.cc`, `src/group.cc`, `src/init.cc`

## Requirements

- Clean NCCL source tree matching the version you want (tarball or git tag)
- CUDA toolkit + compatible host compiler
- Nsight Systems (`nsys`) available in the same environment you will **export** traces from
- Correct GPU target(s) for your system (set `NVCC_GENCODE`). Examples:
   - H100/GH200 (SM90): `-gencode=arch=compute_90,code=sm_90`
   - A100 (SM80): `-gencode=arch=compute_80,code=sm_80`

## Build steps

Choose one of the supported version paths below.

### Option A: NCCL 2.28.3-1 (apply patch)

#### 1) Apply the patch

Apply the patch shipped in this directory to a clean NCCL `2.28.3-1` source tree.

If you don’t already have the sources, the easiest way is to clone the matching tag:

```bash
git clone https://github.com/NVIDIA/nccl.git -b v2.28.3 $SCRATCH/nccl_228_npkit/nccl
```

```bash
cd $SCRATCH/nccl_228_npkit/nccl
```

Then apply the patch:

```bash
git apply <repo_root>/goal_gen/ai/nccl_versions/nccl_atlahs_228.patch
```

Alternatively, if you start from a tarball:

```bash
tar xf nccl-2.28.3-1.tar.gz
cd nccl-2.28.3-1
git apply <repo_root>/goal_gen/ai/nccl_versions/nccl_atlahs_228.patch
```

#### 2) Build NCCL with NVTX enabled

```bash
make clean

export TRACING_FLAGS="-DENABLE_API_NVTX -DENABLE_INIT_NVTX -DENABLE_ENQUEUE_NVTX"
export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"  # adjust for your GPU

make -j src.build \
   CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
   NVCC_GENCODE="$NVCC_GENCODE" \
   TRACING_FLAGS="$TRACING_FLAGS"
```

Result: `build/lib/libnccl.so` (NVTX-enabled).

#### 3) Use the built library

For NCCL 2.28, prefer `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/nccl-2.28.3-1/build/lib:"$LD_LIBRARY_PATH"
```

Use `LD_PRELOAD` only if you must override a bundled/system NCCL:

```bash
export LD_PRELOAD=/path/to/nccl-2.28.3-1/build/lib/libnccl.so
```

Avoid setting both unless you have a specific reason.

### Option B: NCCL 2.20.x (annotated source drop)

Historically, ATLAHS started from an NCCL 2.20-based annotated source tree.

In this repo, NCCL 2.20.5 annotated source is provided as a **git submodule** at:

- `goal_gen/ai/nccl_versions/nccl_nvtx_v2.20.5/`

Upstream repo + pinned commit:

- https://github.com/ZhiyiHu1999/nccl_nvtx_v2.20.5-1/tree/afd5dc513dfc79822da1f97fa6cf557a90902a4c

If you cloned this repo without submodules, initialize it with:

```bash
git submodule update --init --recursive
```

Then build the 2.20.5 tree like a normal NCCL release, keeping `TRACING_FLAGS` enabled.

## Collect traces (nsys)

Run your workload under `nsys` with NVTX enabled (you can use any launcher: direct run, mpirun, srun, sbatch, etc.).

If you **late-capture** with `--capture-range=cudaProfilerApi`, the NCCL INIT NVTX markers
(ring/tree/comm init) will be missing from the SQLite output because NCCL initializes
before `cudaProfilerStart()`. In that case, also enable NCCL INFO logs so GOAL generation
can reconstruct the topology from text logs:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_DEBUG_FILE=/path/to/nccl_logs/nccl_%h_%p.log
```

Important: the NCCL 2.28 patch currently emits NVTX ranges in the **default NVTX domain** (i.e., it does not create/use a named `NCCL` domain). If you run `nsys` with `--nvtx-domain-include=NCCL`, you will filter these ranges out and ATLAHS will not see them. Omit `--nvtx-domain-include` unless you also change the patch to emit into a named domain.

Typical flags that work well with ATLAHS post-processing:

```bash
nsys profile \
   --trace=nvtx,cuda \
   --cuda-memory-usage=false \
   --cuda-um-cpu-page-faults=false \
   --cuda-um-gpu-page-faults=false \
   -s none \
   -o <trace_dir>/nsys_report \
   <your_command_here>
```

Both layouts work:

- **Per-node** reports (historical NCCL 2.20 ETH traces): one `.nsys-rep` containing multiple ranks/GPUs.
- **Per-rank/GPU** reports (common NCCL 2.28): one `.nsys-rep` per rank. If you export to SQLite, prefer filenames like `profile_<jobid>_<node>_<rank>.sqlite` so the goal generator can infer rank↔host mapping when comm-init markers are missing.

If you are using the repository scripts, you can adapt:

- `apps/ai/scripts/run_megatron.sh`
- `apps/ai/scripts/run_vllm.sh`

## Export traces to SQLite

Preferred: use the helper script (from the repo root):

```bash
bash scripts/nsys_reports_to_sqlite.sh <trace_dir>
```

If you hit an `nsys export` version mismatch, run export using the same container/module/environment that created the `.nsys-rep` files.

## Quick NVTX sanity check

Pick one exported SQLite file and verify NCCL NVTX ranges exist:

```bash
python3 - <<'PY'
import os
import sqlite3

db = os.environ.get("NSYS_SQLITE", "<trace_dir>/nsys_report.sqlite")
con = sqlite3.connect(db)
cur = con.cursor()

cur.execute("select count(*) from NVTX_EVENTS where text like 'nccl%'")
print("NCCL NVTX:", cur.fetchone()[0])

cur.execute("select count(*) from NVTX_EVENTS where text like '%nWarps % sendbuff % recvbuff %'")
print("Enqueue markers:", cur.fetchone()[0])
PY
```

Alternative quick check (no export needed):

```bash
strings <trace_dir>/*.nsys-rep | grep -m1 nWarps
```

## GOAL → BIN → LGS

1) Generate GOAL events from traces:

```bash
python3 goal_gen/ai/nccl_goal_generator/get_traced_events.py \
   -i <trace_dir>/ -o <goal_dir>/ --unique-nic --merge-non-overlap \
   -l goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/<platform>/npkit_data_summary_LL.json \
   -s goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/<platform>/npkit_data_summary_Simple.json
```

If INIT NVTX is missing (late capture), add NCCL log reconstruction:

```bash
python3 goal_gen/ai/nccl_goal_generator/get_traced_events.py \
   -i <trace_dir>/ -o <goal_dir>/ --unique-nic --merge-non-overlap \
   -l goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/<platform>/npkit_data_summary_LL.json \
   -s goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/<platform>/npkit_data_summary_Simple.json \
   --nccl-debug-log-dir /path/to/nccl_logs \
   --use-nccl-debug-topology
```

`<platform>` should match one of the available directories under `goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/` (e.g., `ault`, `clariden`, `play`).

2) Convert text GOAL → binary:

```bash
sim/LogGOPSim/txt2bin \
   -i <goal_dir>/InterNode_MicroEvents_Dependency.goal \
   -o <goal_dir>/<name>.bin
```

3) Run LogGOPSim:

```bash
sim/LogGOPSim/LogGOPSim \
   -f <goal_dir>/<name>.bin -L 3700 -o 200 -g 5 -O 0 -G 0.04 -S 0 \
   | tee <goal_dir>/<name>.lgs.log
```

## Troubleshooting

- **No NVTX markers**: ensure `TRACING_FLAGS` were set during the NCCL build and that your run is actually loading the patched `libnccl.so`.
- **Wrong NCCL picked up**: check your environment (`LD_LIBRARY_PATH` / `LD_PRELOAD`) and verify with `ldd <your_binary> | grep nccl`.
- **`nsys export` mismatch**: export using the same Nsight Systems version/environment that generated the `.nsys-rep`.
- **Empty globs in scripts**: enable `nullglob` in bash before looping over `*.nsys-rep`:

   ```bash
   shopt -s nullglob
   ```

## Adding support for more NCCL versions

To extend support, add a new patch (recommended) or an annotated source directory under this folder, then update the table above.

Suggested naming:

- Patch: `nccl_atlahs_<major><minor>.patch` (example: `nccl_atlahs_228.patch`)
- Source drop: `nccl_nvtx_v<nccl_version>/`
