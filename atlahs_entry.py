import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="ATLAHS Simulator Entry")
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Specify the simulator backend to use: 'NS-3', 'htsim', or 'LGS'"
    )
    parser.add_argument(
        "--goal_file",
        type=str,
        required=True,
        help="GOAL file describing the workload"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File to save the output information"
    )
    
    args = parser.parse_args()
    backend = args.backend.lower()
    
    if backend == "ns-3":
        # Initialize and run NS-3 simulator
        print("Using NS-3 simulator")
    elif backend == "htsim":
        # Initialize and run htsim simulator
        print("Using htsim simulator")
        cmd = f"./sim/htsim_test/sim/datacenter/htsim_eqds -topo sim/htsim_test/sim/datacenter/topologies/leaf_spine_tiny.topo -tm sim/htsim_test/sim/datacenter/connection_matrices/perm_32n_32c_2MB.cm > {args.output_file}"
        subprocess.run(cmd, shell=True)
    elif backend == "lgs":
        # Initialize and run LGS simulator
        print("Using LGS simulator")
    else:
        print(f"Error: Unknown simulator backend '{args.backend}'")
        sys.exit(1)

if __name__ == "__main__":
    main()