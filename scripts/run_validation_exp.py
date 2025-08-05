import os
import argparse
import sys
from typing import List, Tuple, Dict, Optional


RUN_SIMULATOR_SCRIPT_PATH = "/workspace/scripts/run_simulator.py"

OUTPUT_DIR = "/workspace/data/validation/"

LGS_CONFIG_PATH = "/workspace/scripts/configs/lgs_config.yaml"

LGS_EXEC_PATH = "/workspace/sim/LogGOPSim/LogGOPSim"
# TODO Tommaso: Add htsim executable path
HT_SIM_EXEC_PATH = ""
ASTRA_SIM_EXEC_SCRIPT = "/workspace/apps/ai/astra-sim/example/run_network_analytical.sh"


def print_warning(message: str, flush: bool = True) -> None:
    print(f"\033[93m[WARNING] {message}\033[0m", flush=flush)

def print_error(message: str, flush: bool = True) -> None:
    print(f"\033[91m[ERROR] {message}\033[0m", flush=flush)

def print_success(message: str, flush: bool = True) -> None:  
    print(f"\033[92m[SUCCESS] {message}\033[0m", flush=flush)
    
def print_info(message: str, flush: bool = True) -> None:
    print(f"[INFO] {message}", flush=flush)


def get_workload_dirs(data_dir: str, app_type: str) -> List[str]:
    """
    Fetches the workload directories containing the traces in the given data directory.
    """
    assert app_type in ["ai", "hpc"], "Invalid app type."
    assert os.path.exists(data_dir), "Data directory does not exist."
    
    # Fetches the workload directory containing the traces in the given data directory
    dataset_dir = os.path.join(data_dir, app_type)
    if not os.path.exists(dataset_dir):
        print_error(f"Dataset directory {dataset_dir} does not exist.")
        exit(1)
    
    apps = os.listdir(dataset_dir)
    for app in apps:
        app_dir = os.path.join(dataset_dir, app)
        if not os.path.isdir(app_dir):
            print_error(f"App directory {app_dir} is not a directory.")
            exit(1)

    print_info(f"Apps: {apps}")

    workload_dirs = []
    for app in apps:
        app_dir = os.path.join(dataset_dir, app)
        workloads = os.listdir(app_dir)
        for workload in workloads:
            workload_dir = os.path.join(app_dir, workload)
            workload_dirs.append(workload_dir)
    
    print_info(f"Workload directories: {workload_dirs}")
    return workload_dirs


# Get astrasim workload dirs
def get_astrasim_workload_dirs(data_dir: str) -> List[str]:
    """
    Fetches the workload directories containing the traces in the given data directory.
    """
    assert os.path.exists(data_dir), "Data directory does not exist."
    
    # Fetches the workload directory containing the traces in the given data directory
    dataset_dir = os.path.join(data_dir, "astrasim")
    if not os.path.exists(dataset_dir):
        print_error(f"Dataset directory {dataset_dir} does not exist.")
        exit(1)
    
    workload_dirs = []
    workload_names = os.listdir(dataset_dir)
    for workload_name in workload_names:
        workload_dir = os.path.join(dataset_dir, workload_name)
        if not os.path.isdir(workload_dir):
            print_error(f"Workload directory {workload_dir} is not a directory.")
            exit(1)
        workload_dirs.append(workload_dir)
    return workload_dirs

def run_validation_exp(data_dir: str, app_type: str, verbose: bool) -> None:
    """
    Runs the validation experiment for the given data directory.
    """
    print_info(f"Running the validation experiment for {app_type.upper()} workloads...")

    workload_dirs = get_workload_dirs(data_dir, app_type)
    astrasim_workload_dirs = get_astrasim_workload_dirs(data_dir) if app_type == "ai" else []
    
    for workload_dir in workload_dirs:
        output_dir = os.path.join(OUTPUT_DIR, app_type)
        lgs_cmd = f"python3 {RUN_SIMULATOR_SCRIPT_PATH} -i {workload_dir} -o {output_dir} -v -c {LGS_CONFIG_PATH} -s atlahs_lgs -t {app_type} --exec {LGS_EXEC_PATH}"
        print_info(f"Running command: {lgs_cmd}")
        assert os.system(lgs_cmd) == 0, f"Error running the LGS simulator for {workload_dir}."

        htsim_cmd = "" # TODO Tommaso: Add htsim command

        # astrasim_cmd = f"bash {ASTRA_SIM_EXEC_SCRIPT} -i {workload_dir} -o {output_dir} -v -c {ASTRA_SIM_CONFIG_PATH} -s astra_sim -t {app_type} -e {ASTRA_SIM_EXEC_PATH}"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the validation experiment for the paper.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory containing the data.')
    parser.add_argument('-t', '--app-type', type=str, default="ai",
                        help="Type of application traces to be simulated. Options are 'hpc' and 'ai'.")

    args = parser.parse_args()

    run_validation_exp(args.data_dir, args.app_type, verbose=True)
