# A helper script that wraps around chakra_trace_link and chakra_converter
# to convert Pytorch execution traces to Chakra traces.
# Reference:
# https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces

import os
import argparse
import subprocess
from time import time


def link_execution_traces(input_dir: str, output_dir: str, host_et_name: str, device_et_name: str) -> int:
    """
    Links the host and device execution traces for each rank
    Args:
        input_dir: Input directory containing Pytorch execution traces
        output_dir: Output directory to store the linked execution traces
        host_et_name: Host ET file name pattern
        device_et_name: Device ET file name pattern
    Returns:
        num_ranks: Number of ranks detected
    """
    # Automatically detect the number of ranks from the input directory
    num_ranks = 0
    while True:
        if os.path.exists(os.path.join(input_dir, host_et_name.replace("*", str(num_ranks)))):
            num_ranks += 1
        else:
            break
    
    if num_ranks == 0:
        print("[INFO] No ranks detected in the input directory")
        exit(0)

    abs_output_dir = os.path.abspath(output_dir)
    assert os.path.exists(output_dir), f"[ERROR] Output directory {output_dir} does not exist"

    print(f"[INFO] Detected {num_ranks} ranks")
    current_dir = os.getcwd()
    # Links all the host and device ET files
    # Cd into the input directory
    os.chdir(input_dir)
    for rank in range(num_ranks):
        # host_et = os.path.join(input_dir, host_et_name.replace("*", str(rank)))
        # device_et = os.path.join(input_dir, device_et_name.replace("*", str(rank)))
        host_et = host_et_name.replace("*", str(rank))
        device_et = device_et_name.replace("*", str(rank))

        link_et = os.path.join(abs_output_dir, f"rank{rank}_link.json")
        cmd = ["chakra_trace_link", "--rank", str(rank), "--chakra-host-trace", host_et, "--chakra-device-trace", device_et, "--output-file", link_et]
        print(f"[DEBUG] Running command: {' '.join(cmd)}")
        r = subprocess.run(cmd, check=True)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to link execution traces for rank {rank}")
        print(f"[INFO] Successfully linked execution traces for rank {rank}")
    print("[INFO] Successfully linked all execution traces")
    os.chdir(current_dir)
    return num_ranks


def convert_linked_traces(num_ranks: int, input_dir: str, output_dir: str, remove_intermediate: bool) -> None:
    """
    Converts the linked execution traces to Chakra traces
    Args:
        num_ranks: Number of ranks
        input_dir: Input directory containing Pytorch execution traces
        output_dir: Output directory to store the Chakra traces
    """
    print(f"[INFO] Converting linked execution traces to Chakra traces")
    for rank in range(num_ranks):
        link_et = os.path.join(output_dir, f"rank{rank}_link.json")
        chakra_et = os.path.join(output_dir, f"chakra.{rank}.et")
        cmd = ["chakra_converter", "PyTorch", "--input", link_et, "--output", chakra_et]
        r = subprocess.run(cmd, check=True)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to convert linked execution traces for rank {rank}")
        print(f"[INFO] Successfully converted linked execution traces for rank {rank}")
        if remove_intermediate:
            os.remove(link_et)
            print(f"[INFO] Removed intermediate linked execution trace for rank {rank}")
    print("[INFO] Successfully converted all linked execution traces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pytorch execution traces to Chakra traces")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Input directory containing Pytorch execution traces")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory to store Chakra traces")
    parser.add_argument("--host-et-name", type=str, default="rank*_host.json", help="Host ET file name pattern")
    parser.add_argument("--device-et-name", type=str, default="rank*_device.json", help="Device ET file name pattern")
    parser.add_argument("-r", "--remove-intermediate", action="store_true", help="Remove intermediate linked traces")

    args = parser.parse_args()
    
    
    assert os.path.exists(args.input_dir), f"[ERROR] Input directory {args.input_dir} does not exist"
    if os.path.exists(args.output_dir):
        print(f"[INFO] Output directory {args.output_dir} already exists. Deleting the directory")
        os.system(f"rm -rf {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time()
    # Step 1: Link Pytorch traces to produce a single file for host and device
    num_ranks = link_execution_traces(args.input_dir, args.output_dir, args.host_et_name, args.device_et_name)

    # Step 2: Convert the linked traces to Chakra traces
    convert_linked_traces(num_ranks, args.input_dir, args.output_dir, args.remove_intermediate)
    print(f"[INFO] Total time taken: {time() - start_time:.2f} seconds")
    print("[INFO] Conversion completed successfully")
    