# NCCL Demo

Minimal ATLAHS LLAMA/NCCL demo for the 2-node LLAMA 7B case.

Only the traced `sbatch` step is cached. The login-node stages are meant to be shown live and regenerate their outputs under `runs/llama7b_n2/`.

1. `./01_submit_megatron.sh`
   Prints or validates the 2-node `sbatch` command. Real runs use `srun --environment=megatron` on `debug`.
2. `./02_export_sqlite.sh`
3. `./03_generate_goal_v2.sh`
4. `./04_run_loggopsim.sh`
5. `./05_run_htsim.sh`
6. `./06_show_summary.sh`

For the live demo, `cache/trace_job/raw_nsys/` is pre-seeded from the local LLAMA evaluation run and stands in for the already-completed traced job. SQLite export, V2 GOAL generation, LogGOPSim, and HTSIM are rerun locally into `runs/llama7b_n2/` each time.

The demo uses LGS with `-L 3700 -o 200 -g 5 -O 0 -G 0.04 -S 0` and HTSIM with a demo-local 8-node, 200 Gbps leaf-spine topology under `topologies/leaf_spine_8_1os.topo`.
