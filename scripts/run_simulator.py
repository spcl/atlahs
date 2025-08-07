import os
import argparse
import pandas as pd
from tqdm import tqdm
import yaml
import re
import subprocess
from pathlib import Path

from typing import List, Dict, Optional, Tuple

# Path to the schedgen executable
SCHEDGEN_EXEC_PATH = "/workspace/goal_gen/hpc/Schedgen/schedgen"
# Path to the txt2bin executable
TXT2BIN_EXEC_PATH = "/workspace/sim/LogGOPSim/txt2bin"

# NCCL_GOAL_GEN_PATH
NCCL_GOAL_GEN_PATH = "/workspace/goal_gen/ai/nccl_goal_generator/get_traced_events.py"
NPKIT_SIMPLE_PATH = "/workspace/goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/clariden/npkit_data_summary_Simple.json"
NPKIT_LL_PATH = "/workspace/goal_gen/ai/nccl_goal_generator/npkit_benchmark_results/clariden/npkit_data_summary_LL.json"

# Path to the AstraSim system configuration file
ASTRA_SIM_SYSTEM_PATH = "/workspace/scripts/configs/astrasim_sys.json"

# ===============================================
# Utility functions
# ===============================================

def print_warning(message: str, verbose: bool = True, flush: bool = True) -> None:
    """
    Prints a warning message in color orange.
    """
    if verbose:
        CSTART = '\033[93m'
        CEND = '\033[0m'
        print(f"{CSTART}[WARNING] {message}{CEND}", flush=flush)


def print_error(message: str, verbose: bool = True, flush: bool = True) -> None:
    """
    Prints an error message in color red.
    """
    if verbose:
        CSTART = '\033[91m'
        CEND = '\033[0m'
        print(f"{CSTART}[ERROR] {message}{CEND}", flush=flush)


def print_success(message: str, verbose: bool = True, flush: bool = True) -> None:
    """
    Prints a success message in color green.
    """
    if verbose:
        CSTART = '\033[92m'
        CEND = '\033[0m'
        print(f"{CSTART}[SUCCESS] {message}{CEND}", flush=flush)


def print_info(message: str, verbose: bool = True, flush: bool = True) -> None:
    """
    Prints an information message in color blue.
    """
    if verbose:
        print(f"[INFO] {message}", flush=flush)


def check_dir_exists(directory: str, verbose: bool) -> None:
    """
    A utility function to check if the given directory exists and if it should be overwritten.
    """
    if os.path.exists(directory):
        print_info(f"Directory {directory} already exists. Continuing without overwriting.", verbose)
    else:
        os.makedirs(directory, exist_ok=True)
        print_info(f"Created directory {directory}.", verbose)


def write_result_to_csv(pred_runtime: int, result_dir: str,
                        app_type: str, simulator: str, workload_name: str) -> None:
    """
    A helper function to write the results to CSV files in the
    result directory.
    """
    assert os.path.exists(result_dir), f"Result directory {result_dir} does not exist."
    assert app_type in ["ai", "hpc"], "Invalid app type."
    assert simulator in ["atlahs_lgs", "atlahs_htsim", "astra_sim"], "Invalid simulator."

    output_file = os.path.join(result_dir, f"{simulator}.csv")
    with open(output_file, "a") as f:
        f.write(f"{workload_name},{pred_runtime}\n")
    print_success(f"Wrote the predicted runtime to {output_file}")



def write_results_to_csv(data: str, result_file: str, verbose: bool) -> None:
    """
    Write the results to a CSV file.
    """
    with open(result_file, 'a') as f:
        f.write(data + "\n")
    print_success(f"Results written to {result_file}.", verbose)

            

def get_pred_runtime_for_lgs_simulator(output: str, verbose: bool) -> int:
    """
    Parse the output of the LGS simulator to get the predicted runtime.
    """
    pattern = r"[hH]ost \d+: (\d+)"
    
    match = re.search(pattern, output)
    assert match is not None, "Error parsing the output of the LGS simulator."
    # Iterates over the matches and returns the maximum predicted runtime
    return max([int(m) for m in match.groups()])


# ===============================================
# Main functions
# ===============================================

def run_lgs_simulator(bin_file: str, sim_config: str, exec: str, verbose: bool) -> int:
    """
    Run the LGS simulator for the given binary file and configuration file.
    @return: The predicted runtime of the application in ns.
    """
    assert os.path.exists(bin_file), f"Binary file {bin_file} does not exist."
    assert os.path.exists(sim_config), f"Configuration file {sim_config} does not exist."
    assert os.path.exists(exec), f"Executable {exec} does not exist."

    # Reads the configuration file in yaml format
    with open(sim_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Fetches the LogGOPS parameters from the configuration file
    L = config.get("L", 5000)
    o = config.get("o", 250)
    g = config.get("g", 5)
    G = config.get("G", 0.04)
    O = config.get("O", 0)
    S = config.get("S", 0)

    print(f"Running LGS simulator with parameters: L={L}, o={o}, g={g}, G={G}, O={O}, S={S}")
    cmd = f"{exec} -b {bin_file} -L {L} -o {o} -g {g} -G {G} -O {O} -S {S} -f {bin_file}"
    print_info(f"Running command: {cmd}", verbose)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    assert process.returncode == 0, f"Error running the LGS simulator: {err.decode()}"
    print_info(f"Output: {out.decode()}", verbose)

    # Parse the output to get the predicted runtime
    pred_runtime = get_pred_runtime_for_lgs_simulator(out.decode(), verbose)
    return pred_runtime


def run_astrasim_simulator(workload_dir: str, network_config: str, exec: str, verbose: bool) -> int:
    """
    Run the AstraSim simulator for the given binary file and configuration file.
    @return: The predicted runtime of the application in ns.
    """
    assert os.path.exists(workload_dir), f"Workload directory {workload_dir} does not exist."
    assert os.path.exists(network_config), f"Network configuration file {network_config} does not exist."
    assert os.path.exists(exec), f"Executable {exec} does not exist."
    
    # Counts the number of files in the workload directory
    npu_count = len(os.listdir(workload_dir))
    print_info(f"Number of NPUs: {npu_count}", verbose)

    # Modidfy the content of the network configuration file to specify the number of NPUs
    # Note that the configuration file is a yaml file
    with open(network_config, 'r') as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    content["npus_count"] = [4, npu_count // 4]
    with open(network_config, 'w') as f:
        yaml.dump(content, f)

    # Sets the environment variables for the AstraSim simulator
    os.environ["ASTRA_SIM_WORKLOAD"] = workload_dir + "/chakra"
    os.environ["ASTRA_SIM_NETWORK"] = network_config
    os.environ["ASTRA_SIM_SYSTEM"] = ASTRA_SIM_SYSTEM_PATH

    cmd = f"bash {exec}"
    print_info(f"Running command: {cmd}", verbose)
    # assert os.system(cmd) == 0, f"Error running the AstraSim simulator."
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    assert process.returncode == 0, f"Error running the AstraSim simulator: {err.decode()}"
    # Iterates through the output and find the pattern "finished, (\d+) cycles",
    # and returns the maximum number of cycles found in the output
    pattern = r"finished, (\d+) cycle"
    # Finds all the matches in the output
    matches = re.findall(pattern, out.decode())
    assert len(matches) > 0, "Error parsing the output of the AstraSim simulator."
    pred_runtime = int(max([int(m) for m in matches]))
    return pred_runtime


def run_validation_exp_for_workload(workload_dir: str, result_dir: str, simulator: str,
                                    sim_config: str, exec: str, app_type: str,
                                    verbose: bool) -> int:
    """
    Run the validation experiment for the given workload directory and store the results
    in the result directory.
    :param workload_dir: Directory containing the workload traces.
    :param result_dir: Directory to store the results.
    :param simulator: Simulator to use for the validation experiment.
    :param sim_config: Configuration file for the given simulator.
    :param exec: Executable to use for the simulator.
    :param app_type: Type of application traces to be simulated.
    :param verbose: Print verbose output.
    @#return: Tuple containing two items, where the first item is the
    actual application runtime and second item is the predicted runtime.
    """
    assert os.path.exists(workload_dir), f"Workload directory {workload_dir} does not exist."

    if simulator == "astra_sim":
        pred_runtime = run_astrasim_simulator(workload_dir, sim_config, exec, verbose)
        return pred_runtime
    
    # Retrieves the trace directory and bin file for the given workload directory by
    # iterating over the subdirectories in the workload directory
    trace_dir = None
    bin_file = None
    for f in os.listdir(workload_dir):
        if f.endswith(".bin"):
            bin_file = os.path.join(workload_dir, f)
        if os.path.isdir(os.path.join(workload_dir, f)):
            trace_dir = os.path.join(workload_dir, f)
    
    assert trace_dir is not None, f"No trace directory found in {workload_dir}."
    print_info(f"Found trace directory {trace_dir} and bin file {bin_file} inside {workload_dir}", verbose)
    assert bin_file is not None, f"No bin file found in {workload_dir}."
    # Run the simulator to get the predicted runtime
    
    pred_runtime = None
    if simulator == "atlahs_lgs":
        pred_runtime = run_lgs_simulator(bin_file, sim_config, exec, verbose)
    elif simulator == "atlahs_htsim":
        # TODO TOMMASO: Implement the function to run the HTSIM simulator
        pass
    else:
        raise ValueError("Invalid simulator.")
    
    assert pred_runtime is not None, "Predicted runtime not found."
    print(f"[INFO] Predicted runtime: {pred_runtime / 1e9:.3f} s")
    return pred_runtime


def convert_raw_traces_to_bin_for_hpc(workload_dir: str, workload_name: str, verbose: bool) -> None:
    """
    Convert the raw traces into bin file for the HPC traces.
    """
    assert os.path.exists(SCHEDGEN_EXEC_PATH), f"Schedgen executable {SCHEDGEN_EXEC_PATH} does not exist. Build the schedgen executable first."
    trace_dir = os.path.join(workload_dir, "mpi_traces")
    goal_file_path = os.path.join(workload_dir, f"{workload_name}.goal")
    bin_file_path = os.path.join(workload_dir, f"{workload_name}.bin")
    rank_file_path = os.path.join(trace_dir, "pmpi-trace-rank-0.txt")
    assert os.path.exists(rank_file_path), f"PMPI trace file {rank_file_path} does not exist."
    # Checks if the bin file already exists
    if os.path.exists(bin_file_path):
        print_info(f"bin file {bin_file_path} already exists. Skipping bin file generation.", verbose)
        return
    
    # Checks if the GOAL file already exists, if it not, run the schedgen executable to convert the raw traces into GOAL file
    if not os.path.exists(goal_file_path):
        cmd = f"{SCHEDGEN_EXEC_PATH} -p trace --traces {rank_file_path} -o {goal_file_path}"
        print_info(f"Running command: {cmd}", verbose)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        assert process.returncode == 0, f"Error running the schedgen: {err.decode()}"
        print_info(f"Output: {out.decode()}", verbose)
        print_success(f"Successfully converted the MPI traces into GOAL file for {workload_name}.", verbose)

    assert os.path.exists(TXT2BIN_EXEC_PATH), f"txt2bin executable {TXT2BIN_EXEC_PATH} does not exist. Build the txt2bin executable first."
    bin_file_path = os.path.join(workload_dir, f"{workload_name}.bin")
    cmd = f"{TXT2BIN_EXEC_PATH} -i {goal_file_path} -o {bin_file_path}"
    print_info(f"Running command: {cmd}", verbose)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    assert process.returncode == 0, f"Error running the txt2bin: {err.decode()}"
    print_success(f"Successfully converted the GOAL file into bin file for {workload_name}.", verbose)


def convert_raw_traces_to_bin_for_ai(workload_dir: str, workload_name: str, verbose: bool) -> None:
    """
    Convert the raw traces into bin file for the AI traces.
    """
    assert os.path.exists(NCCL_GOAL_GEN_PATH), f"NCCL goal generator executable {NCCL_GOAL_GEN_PATH} does not exist."
    trace_dir = os.path.join(workload_dir, "nsys_reports")
    bin_file_path = os.path.join(workload_dir, f"{workload_name}.bin")
    goal_file_path = os.path.join(workload_dir, f"InterNode_MicroEvents_Dependency.goal")
    if os.path.exists(bin_file_path):
        print_info(f"bin file {bin_file_path} already exists. Skipping bin file generation.", verbose)
        return
    
    if not os.path.exists(goal_file_path):
        cmd = f"python3 {NCCL_GOAL_GEN_PATH} -i {trace_dir} -o {workload_dir} -q --unique-nic --merge-non-overlap -l {NPKIT_LL_PATH} -s {NPKIT_SIMPLE_PATH}"
        print_info(f"Running command: {cmd}", verbose)
        assert os.system(cmd) == 0, f"Error running the nccl goal generator."
        print_success(f"Successfully converted the raw traces into GOAL file for {workload_name}.", verbose)
    
    cmd = f"{TXT2BIN_EXEC_PATH} -i {goal_file_path} -o {bin_file_path}"
    print_info(f"Running command: {cmd}", verbose)
    assert os.system(cmd) == 0, f"Error running the txt2bin."
    print_success(f"Successfully converted the GOAL file into bin file for {workload_name}.", verbose)


def convert_raw_traces_to_bin(trace_dir: str, workload_name: str, verbose: bool, app_type: str) -> None:
    """
    Convert the raw traces into bin file as per the given app type.
    :param trace_dir: Directory containing the traces.
    :param workload_name: Name of the workload.
    :param app_name: Name of the application.
    :param verbose: Print verbose output.
    :param app_type: Type of application traces to be simulated.
    """
    if app_type == "hpc":
        convert_raw_traces_to_bin_for_hpc(trace_dir, workload_name, verbose)
    elif app_type == "ai":
        convert_raw_traces_to_bin_for_ai(trace_dir, workload_name, verbose)
    else:
        raise ValueError("Invalid application type.")



def run_simulator(trace_dir: str, result_dir: str, simulator: str,
                  sim_config: str, exec: str, app_type: str,
                  verbose: bool) -> None:
    """
    Run the validation experiment for the given trace directory and store the results
    in the result directory.
    :param trace_dir: Directory containing the traces.
    :param result_dir: Directory to store the results.
    :param simulator: Simulator to use for the validation experiment.
    :param sim_config: Configuration file for the given simulator.
    :param exec: Executable to use for the simulator.
    :param app_type: Type of application traces to be simulated.
    :param verbose: Print verbose output.
    """
    assert os.path.exists(trace_dir), f"Trace directory {trace_dir} does not exist."
    check_dir_exists(result_dir, verbose)

    workload_name = Path(trace_dir).name
    app_name = Path(trace_dir).name.split("_")[0]
    print_info(f"App: {app_name}, Workload: {workload_name}", verbose)

    if simulator != "astra_sim":
        # Converts the raw traces into bin file
        convert_raw_traces_to_bin(trace_dir, workload_name, verbose, app_type)

    # Run the validation experiment for the given workload
    pred_t = run_validation_exp_for_workload(trace_dir, result_dir, simulator,
                                             sim_config, exec, app_type, verbose)
    print_info(f"{workload_name}, Predicted runtime: {pred_t}", verbose)
    
    # Write the results to CSV files
    write_result_to_csv(pred_t, result_dir, app_type, simulator, workload_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the validation experiment for the paper.')
    parser.add_argument('-i', '--trace-dir', type=str, help='Directory containing the traces.')
    parser.add_argument('-o', '--result-dir', type=str, help='Directory to store the results.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    # TODO TOMMASO: Potentially add htsim configuration to a yaml file
    parser.add_argument('-c', '--config', type=str, default="configs/lgs_hpc_config.yaml",
                        help='Configuration file for the given simulator.')
    parser.add_argument('-t', '--app-type', type=str, default="ai",
                        help="Type of application traces to be simulated. Options are 'hpc' and 'ai'.")
    parser.add_argument('-s', '--simulator', type=str, default="atlahs_lgs",
                        help='Simulator to use for the validation experiment. Options are atlahs_lgs, atlahs_htsim, astra_sim.')
    parser.add_argument('--exec', type=str, required=True,
                        help='Executable to use for the simulator.')
    args = parser.parse_args()

    print_info(f"Running validation experiment:")
    print_info(f"Trace directory: {args.trace_dir}")
    print_info(f"Result directory: {args.result_dir}")
    assert args.simulator in ("atlahs_lgs", "atlahs_htsim", "astra_sim"), "Invalid simulator."
    print_info(f"Simulator: {args.simulator}")
    assert args.app_type in ("hpc", "ai"), "Invalid application type."
    print_info(f"Type of application traces: {args.app_type}")
    print_info(f"Configuration file: {args.config}")
    print_info(f"Executable: {args.exec}")
    
    run_simulator(args.trace_dir, args.result_dir, args.simulator, args.config,
                       args.exec, args.app_type, args.verbose)
    print_success("Simulator completed successfully.")
