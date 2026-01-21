# What we changed (NCCL 2.28 + ATLAHS traces + LogGOPSim)

This note enumerates the **exact files** we changed (or patches we produced) to:

1) make **NCCL 2.28** emit the ATLAHS-compatible NVTX/annotation schema,
2) make the ATLAHS pipeline handle **“one report per GPU / per-rank sqlite”** style Nsight exports,
3) avoid **LogGOPSim (LGS) deadlock / early termination** on the newer trace patterns.

The goal here is reproducibility: if something breaks, you can immediately narrow down *which layer* (NCCL vs trace conversion vs GOAL generation vs LGS).

---

## 1) NCCL 2.28 ATLAHS annotations

### Where the patch is
- Patch file to apply to a clean NCCL 2.28.x checkout:
  - `ai_stuff/atlahs/nccl_228_work/nccl_atlahs.patch`
  - (duplicate copy) `/users/btommaso/nccl_228_nvtx_v2.patch`

### What NCCL files it changes (and why)
The patch touches these NCCL sources (diff headers in `nccl_atlahs.patch`):

- `makefiles/common.mk`
  - **Why**: ensure `TRACING_FLAGS` propagate into **both** host compilation (`CXXFLAGS`) and device compilation (`NVCUFLAGS`), so the `ENABLE_*_NVTX` code paths are actually compiled in.
  - Without this, you can “apply the patch” but still get no NVTX ranges/marks in traces.

- `src/collectives.cc`
  - **Why**: add NVTX API-level ranges for NCCL collectives so ATLAHS has stable, parseable “entry” markers (e.g., `ncclAllReduce()...`, includes commHash/stream/data_size/type_size/op/root/peer/pid).

- `src/enqueue.cc`
  - **Why**: add enqueue-level NVTX markers (`CollInfo` / `WorkElemColl`) that contain the fields ATLAHS extracts (e.g., `nWarps`, `chunkCount`, offsets, send/recv buffer addresses, `pid`).
  - This is critical because the GOAL generator uses these strings to infer protocols/algorithms/chunking and to build micro-events.

- `src/group.cc`
  - **Why**: mark group start/end and launch boundaries so the trace has stable markers around grouped launches (important for ordering and for “profiling window” alignment).

- `src/init.cc`
  - **Why**: emit communicator + channel topology metadata (rings/trees/channels, commId, rank/nranks, pid), which the GOAL generator uses to build the mapping of ranks and channels.

### Runtime selection (2.28 vs older)
These aren’t NCCL source changes, but they were necessary to make “NCCL 2.28 + Nsight export” behave reliably:

- For 2.28 we generally prefer **`LD_LIBRARY_PATH=/path/to/nccl/build/lib`** and avoid `LD_PRELOAD`.
  - **Why**: we repeatedly observed `nsys export` failing when `LD_PRELOAD` was used with the 2.28 build/toolchain combination.
  - This is documented in: `ai_stuff/atlahs/nccl_228_work/nccl_parity_detailed.md` and `ai_stuff/atlahs/nccl_228_work/nccl_228_vs_220_differences.md`.

---

## 2) Traces exported as “one report per GPU” / per-rank sqlite

This is about the trace *format/layout* and how we ingest it; it’s not about NCCL.

### Trace conversion: `.nsys-rep` vs `.qdrep`/`.qdstrm` vs direct sqlite
- Updated script:
  - `ai_stuff/atlahs/scripts/nsys_reports_to_sqlite.sh`

**What changed**
- It now scans for `*.nsys-rep`, `*.qdrep`, and `*.qdstrm` in a directory and converts each to `*.sqlite`.
- It now has explicit `NSYS_BIN` discovery and a clear error if `nsys` is missing.

**Why**
- On some setups/versions, Nsight produces `.qdstrm`/`.qdrep` in addition to (or instead of) `.nsys-rep`.
- We needed a single conversion entrypoint that works across those variants (and across “container nsys” vs “host nsys”).

### GOAL generator ingestion: per-node sqlite vs per-rank sqlite
- Updated module:
  - `ai_stuff/atlahs/goal_gen/ai/nccl_goal_generator/generator_modules/nsys_events.py`

**What changed (high level)**
- Added logic to handle 2.28-style per-rank sqlite exports with filenames like:
  - `profile_<jobid>_<node>_<rank>.sqlite`
- Added optional deterministic ordering:
  - `ATLAHS_SORT_NSYS_FILES=1` (default) sorts directory file iteration so runs are reproducible.
- Added “best-effort” inference when comm-init NVTX markers are missing (profiling starts late):
  - infer `nranks` from filenames,
  - infer hostnames via `TARGET_INFO_SYSTEM_ENV` (`Hostname`) when present,
  - restrict “pid -> gpuId” mapping to NCCL-relevant NVTX messages to avoid assigning unrelated PIDs.
- Added start/end slack knobs:
  - `ATLAHS_PROFILE_START_SLACK_NS`, `ATLAHS_PROFILE_END_SLACK_NS`

**Why**
- With per-rank sqlite, the original assumptions (“one file contains all GPUs on a node; comm-init NVTX always present”) stop holding.
- Without these adjustments, parsing can fail early (e.g., missing stream mappings like `'(nil)'`) or can silently mis-map ranks/streams, which then produces wrong GOAL dependencies.

---

## 3) Prevent LGS deadlock / early termination (GOAL correctness)

This is the “logic” part: LGS itself will only execute what the GOAL dependency graph permits.
If the GOAL graph is incomplete or cyclic/unschedulable, LGS can:
- deadlock/hang, or
- “finish” early after executing only a tiny prefix (symptom: very small `Events:` count in LGS output).

### Ring AllReduce dependency correctness (j=0 stage)
Patch file (historical debugging fix, documents the issue clearly):
- `ai_stuff/atlahs/nccl_228_work/atlahs_goal_generator_fix1_ring_allreduce_j0.patch`

Files changed by that patch:
- `ai_stuff/atlahs/goal_gen/ai/nccl_goal_generator/generator_modules/data_dependency_modules/inter_node_dependency.py`

**What/Why**
- For ring AllReduce, some tunings require emitting the `j=0` stage of the RecvCopySend / RecvReduceSend loops.
- If `j=0` is omitted, some ranks end up with unmatched send/recv micro-event lists, which then corrupts downstream “pair by index” constraints and can lead to early LGS termination.

### Intra-node “recv requires send” constraints (can induce deadlocks)
Patch file (opt-out default):
- `ai_stuff/atlahs/nccl_228_work/atlahs_goal_generator_fix2_intra_node_recv_requires_send_default_off.patch`

Files changed by that patch:
- `ai_stuff/atlahs/goal_gen/ai/nccl_goal_generator/generator_modules/data_dependency_modules/inter_node_dependency.py`

**What/Why**
- The generator used to add extra constraints for same-node traffic by pairing send/recv tasks by list index:
  - `recv_task requires send_task`
- With newer traces/tunings (especially ring/simple), per-peer micro-event counts can differ, so “pair by index” can:
  - add invalid edges (wrong pairing),
  - create dependency cycles,
  - and ultimately deadlock LGS.
- The patch makes these constraints **off by default** and controllable via env vars:
  - `ATLAHS_ENABLE_INTRA_NODE_RECV_REQUIRES_SEND=1` (legacy behavior)
  - `ATLAHS_DISABLE_INTRA_NODE_RECV_REQUIRES_SEND=1` (force off)

### Deterministic iteration + key typing fixes (reduce nondeterminism and mismatches)
Files with relevant logic changes (current working tree):
- `ai_stuff/atlahs/goal_gen/ai/nccl_goal_generator/generator_modules/data_dependency_modules/in_gpu_dependency.py`
- `ai_stuff/atlahs/goal_gen/ai/nccl_goal_generator/generator_modules/data_dependency_modules/inter_node_dependency.py`

**What/Why**
- Sort iterations over `gpuId`, `streamId`, and group events by timestamp to avoid dependency graph variation due to dict iteration order.
- Normalize rank-index keys to strings (`str(...)`) where the communicator metadata uses string keys, avoiding KeyError/mismatched lookups.
- Clamp negative gaps (`calc` durations) to zero when timestamp jitter causes small reordering.

### “Latest ATLAHS commit” compatibility for the 2.20 ring/simple traces
We found a regression/behavior difference where **latest** generator code could produce a GOAL graph that made LGS run only ~8k events on our “new” 2.20 traces, while still working on the ETH trace set.

To make a clean/latest ATLAHS checkout work reliably, we used a surgical revert patch (known-good generator modules):
- `/users/btommaso/scratch/atlahs_latest_verify_1768910812/patches/goalgen_revert_in_gpu_and_inter_node_from_e436c1d.patch`

Files it changes:
- `goal_gen/ai/nccl_goal_generator/generator_modules/data_dependency_modules/in_gpu_dependency.py`
- `goal_gen/ai/nccl_goal_generator/generator_modules/data_dependency_modules/inter_node_dependency.py`

**Why**
- On the “new” 2.20 traces, **only** updating internode logic is insufficient: the in-GPU microevent generation and the internode microevent generation need to stay consistent (they share the send/recv micro-event bookkeeping).
- Ablation we ran showed:
  - latest only: LGS `Events=8592` (early termination),
  - only internode: still `Events=8592`,
  - only in_gpu: generator crash,
  - both reverted together: LGS `~5.77M` events and `~2s` max host completion time.

---

## LogGOPSim side note (topology parsing)

This is not directly about “deadlocking”, but it is an actual code change in LGS that removed a runtime dependency:

- `ai_stuff/atlahs/sim/LogGOPSim/Network.hpp`
  - **What**: replaced Graphviz libcgraph DOT parsing with a small internal DOT edge parser (`std::regex`), so LGS does not depend on `libcgraph.so` at runtime.
  - **Why**: simplify portability (especially when running on login nodes / minimal environments).

- `ai_stuff/atlahs/sim/LogGOPSim/Parser.hpp`
  - **What**: widen `GetNumCPU()` / `GetNumNIC()` return types to `uint32_t`.
  - **Why**: avoid truncation when traces/topologies use IDs outside `uint8_t` range.

---

## Quick “what to look at first” when something breaks
- No NVTX markers in traces → start with `ai_stuff/atlahs/nccl_228_work/nccl_atlahs.patch` and the `TRACING_FLAGS` propagation in `makefiles/common.mk`.
- `.qdstrm/.qdrep` or exporter mismatch → start with `ai_stuff/atlahs/scripts/nsys_reports_to_sqlite.sh` (and ensure `nsys export` version matches capture version).
- “per-rank sqlite” / `'(nil)'` stream errors → start with `ai_stuff/atlahs/goal_gen/ai/nccl_goal_generator/generator_modules/nsys_events.py`.
- LGS hangs or finishes after ~thousands of events → start with:
  - ring AllReduce `j=0` logic in `.../inter_node_dependency.py`,
  - intra-node recv->send constraints mode,
  - the known-good revert patch for `in_gpu_dependency.py` + `inter_node_dependency.py`.

