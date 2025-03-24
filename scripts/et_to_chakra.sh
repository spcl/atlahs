#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -i <input_dir> -o <output_dir> [--host-et-name <pattern>] [--device-et-name <pattern>] [-r] [--max-conversion-threads <num>]"
    echo "  -i, --input-dir               Directory containing PyTorch execution traces"
    echo "  -o, --output-dir              Directory where Chakra traces will be stored"
    echo "  --host-et-name                Host ET file pattern (default: 'rank*_host.json')"
    echo "  --device-et-name              Device ET file pattern (default: 'rank*_device.json')"
    echo "  -r, --remove-intermediate     Remove intermediate linked traces after conversion"
    echo "  --max-conversion-threads      Maximum number of concurrent processes for linking and conversion (default: 4)"
    exit 1
}

# Default values
host_et_name="rank*_host.json"
device_et_name="rank*_device.json"
remove_intermediate=false
max_conversion_threads=4

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input-dir)
            input_dir="$2"
            shift; shift
            ;;
        -o|--output-dir)
            output_dir="$2"
            shift; shift
            ;;
        --host-et-name)
            host_et_name="$2"
            shift; shift
            ;;
        --device-et-name)
            device_et_name="$2"
            shift; shift
            ;;
        -r|--remove-intermediate)
            remove_intermediate=true
            shift
            ;;
        --max-conversion-threads)
            max_conversion_threads="$2"
            shift; shift
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
    usage
fi

# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "[ERROR] Input directory '$input_dir' does not exist"
    exit 1
fi

# If the output directory doesn't exist, create it; otherwise, use it as is.
if [ -d "$output_dir" ]; then
    echo "[INFO] Output directory '$output_dir' already exists. Using it as is."
else
    mkdir -p "$output_dir"
fi

# Detect number of ranks by checking for host trace files
num_ranks=0
while [ -f "$input_dir/$(echo "$host_et_name" | sed "s/\*/$num_ranks/")" ]; do
    num_ranks=$((num_ranks+1))
done

if [ "$num_ranks" -eq 0 ]; then
    echo "[INFO] No ranks detected in '$input_dir'"
    exit 0
fi

echo "[INFO] Detected $num_ranks rank(s)"

start_time=$(date +%s)

# Function to control maximum number of concurrent processes
wait_for_slot() {
    while true; do
        running_jobs=$(jobs -p | wc -l)
        if [ "$running_jobs" -lt "$max_conversion_threads" ]; then
            break
        fi
        sleep 0.2
    done
}

# Step 1: Link execution traces concurrently for all ranks (with a limit)
for (( rank=0; rank<num_ranks; rank++ )); do
    link_file="$output_dir/rank${rank}_link.json"
    # Check if the linked file already exists; if so, skip linking for this rank.
    if [ -f "$link_file" ]; then
        echo "[INFO] Linking file for rank $rank already exists, skipping linking process."
        continue
    fi
    wait_for_slot
    host_et=$(echo "$host_et_name" | sed "s/\*/$rank/")
    device_et=$(echo "$device_et_name" | sed "s/\*/$rank/")
    cmd="chakra_trace_link --rank $rank --chakra-host-trace $input_dir/$host_et --chakra-device-trace $input_dir/$device_et --output-file $link_file"
    echo "[DEBUG] Linking rank $rank: $cmd"
    $cmd &
done
wait
echo "[INFO] Successfully linked execution traces for all ranks"

# Step 2: Convert linked traces concurrently for all ranks (with a limit)
for (( rank=0; rank<num_ranks; rank++ )); do
    wait_for_slot
    (
        link_file="$output_dir/rank${rank}_link.json"
        chakra_file="$output_dir/chakra.${rank}.et"
        cmd="chakra_converter PyTorch --input $link_file --output $chakra_file"
        echo "[DEBUG] Converting rank $rank: $cmd"
        $cmd
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to convert linked execution traces for rank $rank"
            exit 1
        fi
        echo "[INFO] Successfully converted linked execution traces for rank $rank"
        if [ "$remove_intermediate" = true ]; then
            rm "$link_file"
            echo "[INFO] Removed intermediate file for rank $rank"
        fi
    ) &
done
wait
echo "[INFO] Successfully converted all linked execution traces"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[INFO] Total time taken: ${elapsed} seconds"
echo "[INFO] Conversion completed successfully"
