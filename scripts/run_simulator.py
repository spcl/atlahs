"""
Runs the validation experiment for the paper by iterating over the directories
containing the traces and running the validation experiment for each trace.
Results will be stored in the results directory as CSV files and plots.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import yaml
import re
import subprocess

from typing import List, Dict, Optional, Tuple

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


def check_dir_exists(directory: str, overwrite: bool, verbose: bool) -> None:
    """
    A utility function to check if the given directory exists and if it should be overwritten.
    """
    if os.path.exists(directory):
        if overwrite:
            print_warning(f"Directory {directory} already exists. Overwriting.", verbose)
            os.removedirs(directory)
            os.makedirs(directory, exist_ok=True)
        else:
            print_info(f"Directory {directory} already exists. Continuing without overwriting.", verbose)
    else:
        os.makedirs(directory, exist_ok=True)
        print_info(f"Created directory {directory}.", verbose)


def write_results_to_csv(results: List[str], result_file: str, verbose: bool) -> None:
    """
    Write the results to a CSV file.
    """
    with open(result_file, 'w') as f:
        f.write("\n".join(results))
    print_success(f"Results written to {result_file}.", verbose)


def get_real_runtime_for_ai_traces(trace_dir: str, verbose: bool) -> float:
    """
    Get the real runtime for the AI traces by subtracting the start time from the end time
    in each of the trace files in the given directory.
    """
    raise NotImplementedError("Function not implemented yet.")


def get_real_runtime_for_hpc_traces(trace_dir: str, verbose: bool) -> int:
    """
    Get the real runtime for the HPC traces by subtracting the start time from the end time
    in each of the trace files in the given directory.
    @return: The real runtime of the application in ns.
    """
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

        return (end_time - start_time) * 1e3
            

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


def run_validation_exp_for_setup(setup_dir: str, result_dir: str, simulator: str,
                                 sim_config: str, exec: str, app_type: str,
                                 overwrite: bool, verbose: bool) -> Tuple[float, int]:
    """
    Run the validation experiment for the given setup directory and store the results
    in the result directory.
    :param setup_dir: Directory containing the setup traces.
    :param result_dir: Directory to store the results.
    :param simulator: Simulator to use for the validation experiment.
    :param sim_config: Configuration file for the given simulator.
    :param exec: Executable to use for the simulator.
    :param app_type: Type of application traces to be simulated.
    :param overwrite: Overwrite existing results.
    :param verbose: Print verbose output.
    @#return: Tuple containing two items, where the first item is the
    actual application runtime and second item is the predicted runtime.
    """
    assert os.path.exists(setup_dir), f"Setup directory {setup_dir} does not exist."
    
    # Retrieves the trace directory and bin file for the given setup directory by
    # iterating over the subdirectories in the setup directory
    trace_dir = None
    bin_file = None
    for f in os.listdir(setup_dir):
        if f.endswith(".bin"):
            bin_file = os.path.join(setup_dir, f)
        if os.path.isdir(os.path.join(setup_dir, f)):
            trace_dir = os.path.join(setup_dir, f)
            break
    
    assert trace_dir is not None, f"No trace directory found in {setup_dir}."
    assert bin_file is not None, f"No bin file found in {setup_dir}."
    print_info(f"Found trace directory {trace_dir} and bin file {bin_file} inside {setup_dir}.", verbose)

    real_runtime = None
    if app_type == "hpc":
        real_runtime = get_real_runtime_for_hpc_traces(trace_dir, verbose)
    elif app_type == "ai":
        real_runtime = get_real_runtime_for_ai_traces(trace_dir, verbose)
    else:
        raise ValueError("Invalid application type.")

    assert real_runtime is not None, "Real runtime not found."
    print(f"[INFO] Real runtime: {real_runtime / 1e9:.3f} s")
    # Run the simulator to get the predicted runtime
    
    pred_runtime = None
    if simulator == "lgs":
        pred_runtime = run_lgs_simulator(bin_file, sim_config, exec, verbose)
    elif simulator == "htsim":
        pass
    elif simulator == "ns3":
        pass
    else:
        raise ValueError("Invalid simulator.")
    
    assert pred_runtime is not None, "Predicted runtime not found."
    print(f"[INFO] Predicted runtime: {pred_runtime / 1e9:.3f} s")
    return real_runtime, pred_runtime

    

def run_simulator(trace_dir: str, result_dir: str, simulator: str,
                       sim_config: str, exec: str, app_type: str,
                       overwrite: bool , verbose: bool) -> None:
    """
    Run the validation experiment for the given trace directory and store the results
    in the result directory.
    :param trace_dir: Directory containing the traces.
    :param result_dir: Directory to store the results.
    :param simulator: Simulator to use for the validation experiment.
    :param sim_config: Configuration file for the given simulator.
    :param exec: Executable to use for the simulator.
    :param app_type: Type of application traces to be simulated.
    :param overwrite: Overwrite existing results.
    :param verbose: Print verbose output.
    """
    assert os.path.exists(trace_dir), f"Trace directory {trace_dir} does not exist."
    check_dir_exists(result_dir, overwrite, verbose)

    # Get the list of subdirectories in the trace directory
    app_dirs = [os.path.join(trace_dir, d) for d in os.listdir(trace_dir) if os.path.isdir(os.path.join(trace_dir, d))]
    
    print_info(f"Found {len(app_dirs)} subdirectories in the trace directory.", verbose)

    # Iterate over the subdirectories and run the validation experiment
    results = ["app_name,setup_name,real_runtime,pred_runtime"]
    for app_dir in app_dirs:
        app_name = os.path.basename(app_dir)
        print_info(f"Running validation experiment for {app_name}.", verbose)

        # Get the list of subdirectories in the app directory for different 
        # setup of the same application
        setup_dirs = [os.path.join(app_dir, d) for d in os.listdir(app_dir) if os.path.isdir(os.path.join(app_dir, d))]
        print_info(f"Found {len(setup_dirs)} subdirectories in the app directory.", verbose)

        # Iterate over the setup directories and run the validation experiment
        for setup_dir in setup_dirs:
            setup_name = os.path.basename(setup_dir)

            # Run the validation experiment for the given setup
            real_t, pred_t = run_validation_exp_for_setup(setup_dir, result_dir, simulator,
                                                          sim_config, exec, app_type, overwrite, verbose)
            print_info(f"{setup_name}, Real runtime: {real_t}, Predicted runtime: {pred_t}", verbose)
            results.append(f"{app_name},{setup_name},{real_t},{pred_t}")
    
    # Write the results to a CSV file
    result_file = os.path.join(result_dir, f"{simulator}_validation_results.csv")
    write_results_to_csv(results, result_file, verbose)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the validation experiment for the paper.')
    parser.add_argument('-i', '--trace-dir', type=str, help='Directory containing the traces.')
    parser.add_argument('-o', '--result-dir', type=str, help='Directory to store the results.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    parser.add_argument('-c', '--config', type=str, default="configs/lgs_config.yaml",
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
                       args.exec, args.app_type, args.overwrite, args.verbose)
    print_success("Simulator completed successfully.")
