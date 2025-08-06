import os
import argparse
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, overload


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


def write_real_runtime_to_file(real_runtime: float, data_dir: str,
                               app_type: str, workload_name: str) -> None:
    """
    A helper function to write the real runtime to the file in the output directory.
    """
    assert os.path.exists(data_dir), "Data directory does not exist."
    assert app_type in ["ai", "hpc"], "Invalid app type."
    
    output_dir = os.path.join(data_dir, "validation", app_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "measured.csv")
    with open(output_file, "a") as f:
        f.write(f"{workload_name},{real_runtime}\n")
    
    print_success(f"Wrote the real runtime to {output_file}")

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


def get_real_runtime_for_hpc_traces(trace_dir: str) -> float:
    """
    Get the real runtime for the HPC traces by subtracting the start time from the end time
    in each of the trace files in the given directory.
    @return: The real runtime of the application in ns.
    """
    assert os.path.exists(trace_dir), f"Trace directory {trace_dir} does not exist."
    
    real_runtime = 0.0
    for trace_file in os.listdir(trace_dir):
        trace_path = os.path.join(trace_dir, trace_file)
        f = open(trace_path, 'r')
        lines = f.readlines()

        # Get the start time
        start_time = None
        end_time = None

        for line in lines:
            if line.startswith("MPI_Init_thread") or line.startswith("MPI_Init"):
                tokens = line.split(":")
                start_time = int(tokens[-1])
                break
        
        for line in reversed(lines):
            if line.startswith("MPI_Finalize"):
                tokens = line.split(":")
                end_time = int(tokens[1])
                break
        
        assert start_time is not None, f"Start time not found in {trace_file}."
        assert end_time is not None, f"End time not found in {trace_file}."

        f.close()

        real_runtime = max(end_time - start_time, real_runtime)

    return real_runtime * 1e3


def get_real_runtime_for_ai_traces(trace_dir: str) -> float:
    """
    Get the real runtime for the AI traces by subtracting the start time from the end time
    in each of the trace files in the given directory.
    @return: The real runtime of the application in ns.
    """
    # FIXME Implement this
    return 0.0


def get_real_runtime_for_workload(workload_dir: str, app_type: str) -> float:
    """
    A helper function to get the real runtime for the workload.
    """
    assert app_type in ["hpc", "ai"], "Invalid app type."
    assert os.path.exists(workload_dir), "Workload directory does not exist."
    
    if app_type == "hpc":
        return get_real_runtime_for_hpc_traces(workload_dir + "/mpi_traces")
    elif app_type == "ai":
        return get_real_runtime_for_ai_traces(workload_dir + "/nsys_reports")
    else:
        raise ValueError("Invalid app type.")
    

def run_validation_exp(data_dir: str, app_type: str, overwrite: bool, verbose: bool) -> None:
    """
    Runs the validation experiment for the given data directory.
    """
    print_info(f"Running the validation experiment for {app_type.upper()} workloads...")

    workload_dirs = get_workload_dirs(data_dir, app_type)
    astrasim_workload_dirs = get_astrasim_workload_dirs(data_dir) if app_type == "ai" else []
    
    if overwrite:
        # Delete the result directory
        result_dir = os.path.join(data_dir, "validation")
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
            print_success(f"Overwriting existing results...")

    for workload_dir in workload_dirs:
        # Get the real runtime for the workload
        real_runtime = get_real_runtime_for_workload(workload_dir, app_type)
        print_info(f"Real runtime for {workload_dir}: {real_runtime} ns")
        # Write the real runtime to the file in the output directory
        workload_name = Path(workload_dir).name
        write_real_runtime_to_file(real_runtime, data_dir, app_type, workload_name)

        # Run the LGS simulator
        output_dir = os.path.join(OUTPUT_DIR, app_type)
        lgs_cmd = f"python3 {RUN_SIMULATOR_SCRIPT_PATH} -i {workload_dir} -o {output_dir} -v -c {LGS_CONFIG_PATH} -s atlahs_lgs -t {app_type} --exec {LGS_EXEC_PATH}"
        print_info(f"Running command: {lgs_cmd}")
        assert os.system(lgs_cmd) == 0, f"Error running the LGS simulator for {workload_dir}."

        htsim_cmd = "" # TODO Tommaso: Add htsim command

        # astrasim_cmd = f"bash {ASTRA_SIM_EXEC_SCRIPT} -i {workload_dir} -o {output_dir} -v -c {ASTRA_SIM_CONFIG_PATH} -s astra_sim -t {app_type} -e {ASTRA_SIM_EXEC_PATH}"

    if app_type == "ai":
        for workload_dir in astrasim_workload_dirs:
            pass
            # output_dir = os.path.join(OUTPUT_DIR, "astrasim")
            # astrasim_cmd = f"bash {ASTRA_SIM_EXEC_SCRIPT} -i {workload_dir} -o {output_dir} -v -c {ASTRA_SIM_CONFIG_PATH} -s astra_sim -t {app_type} -e {ASTRA_SIM_EXEC_PATH}"
            # print_info(f"Running command: {astrasim_cmd}")
            # assert os.system(astrasim_cmd) == 0, f"Error running the AstraSim simulator for {workload_dir}."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the validation experiment for the paper.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory containing the data.')
    parser.add_argument('-t', '--app-type', type=str, default="ai",
                        help="Type of application traces to be simulated. Options are 'hpc' and 'ai'.")
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the existing results.')

    args = parser.parse_args()

    run_validation_exp(args.data_dir, args.app_type, args.overwrite, verbose=True)
