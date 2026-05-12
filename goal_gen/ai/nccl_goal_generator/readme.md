Command for ault:

```
python3 get_traced_events.py \
  --npkit_file_Simple npkit_benchmark_results/ault/npkit_data_summary_Simple.json \
  --npkit_file_LL npkit_benchmark_results/ault/npkit_data_summary_LL.json
```



Command for clariden:

```
python3 get_traced_events.py \
  --npkit_file_Simple npkit_benchmark_results/clariden/npkit_data_summary_Simple.json \
  --npkit_file_LL npkit_benchmark_results/clariden/npkit_data_summary_LL.json
```

Late-capture case (INIT NVTX missing):

```bash
python3 get_traced_events.py \
  --npkit_file_Simple npkit_benchmark_results/clariden/npkit_data_summary_Simple.json \
  --npkit_file_LL npkit_benchmark_results/clariden/npkit_data_summary_LL.json \
  --nccl-debug-log-dir /path/to/nccl_logs \
  --use-nccl-debug-topology
```
