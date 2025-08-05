import os
import sys
import argparse


VALIDATION_SCRIPT_PATH = "/workspace/scripts/run_validation_exp.py"


def print_info(message: str, flush: bool = True) -> None:
    # Print the message in blue
    print(f"\033[94m{message}\033[0m", flush=flush)

def print_warning(message: str, verbose: bool = True, flush: bool = True) -> None:
    # Print the message in yellow
    print(f"\033[93m{message}\033[0m", flush=flush)

def print_error(message: str, flush: bool = True) -> None:
    # Print the message in red
    print(f"\033[91m{message}\033[0m", flush=flush)

def print_success(message: str, flush: bool = True) -> None:
    # Print the message in green
    print(f"\033[92m{message}\033[0m", flush=flush)

STORAGE_SERVER_URL = "http://storage2.spcl.ethz.ch/traces/"
ASTRASIM_URL = STORAGE_SERVER_URL + "astra-sim-traces/"
AI_TRACE_URL = STORAGE_SERVER_URL + "ai/"
HPC_TRACE_URL = STORAGE_SERVER_URL + "hpc/"
CASE_STUDY_URL = STORAGE_SERVER_URL + "case_studies/"


DOWNLOAD_CMD = 'wget -r -np -nH --cut-dirs=4 -R "index.html*" -c -P "{}" "{}"'


AI_TRACES_QUICK_TEST = ["llama/Llama7B_N4_GPU16_TP1_PP1_DP16_BS32", "llama/Llama7B_N32_GPU128_PP1_DP128_7B_BS128"]
ASTRASIM_TRACES_QUICK_TEST = ["llama_N4_GPU16", "llama_N32_GPU128"]
HPC_TRACES_QUICK_TEST = ["lulesh/lulesh_8", "icon/icon_8", "hpcg/hpcg_8"]
CASE_STUDIES_QUICK_TEST = [
    # TODO: TOMMASO Add case studies for the quick test
]


AI_TRACES_FULL_REPRODUCTION = ["llama/Llama7B_N4_GPU16_TP1_PP1_DP16_BS32", "llama/Llama7B_N32_GPU128_PP1_DP128_7B_BS128"]
ASTRASIM_TRACES_FULL_REPRODUCTION = ["llama_N4_GPU16", "llama_N32_GPU128", "llama_N64_GPU256", "MoE_N16_GPU64", "MoE_N32_GPU128", "MoE_N64_GPU256"]
HPC_TRACES_FULL_REPRODUCTION = [
    "lulesh/lulesh_8", "lulesh/lulesh_27", "lulesh/lulesh_64",
    "icon/icon_8", "icon/icon_32", "icon/icon_64",
    "hpcg/hpcg_8", "hpcg/hpcg_32", "hpcg/hpcg_64",
    "lammps/lammps_8", "lammps/lammps_32", "lammps/lammps_64",
    "openmx/openmx_8", "openmx/openmx_32",
    "cloverleaf/cloverleaf_8"
]
CASE_STUDIES_FULL_REPRODUCTION = [
    # TODO: TOMMASO Add case studies for the full reproduction
]


def download_data(data_dir: str, is_quick_test: bool = True) -> None:
    """
    Downloads the necessary data from the storage server based
    on the type of test being run.
    """
    print_info("Downloading the data from the storage server...")
    # Download validation traces
    if is_quick_test:
        # Download the validation traces for AI workloads
        ai_traces = AI_TRACES_QUICK_TEST
        hpc_traces = HPC_TRACES_QUICK_TEST
        case_studies = CASE_STUDIES_QUICK_TEST
    else:
        ai_traces = AI_TRACES_FULL_REPRODUCTION
        hpc_traces = HPC_TRACES_FULL_REPRODUCTION
        case_studies = CASE_STUDIES_FULL_REPRODUCTION
        # Warn the user that this would take a long time and require more than 250 GB of disk space
        print_warning("This would take a long time and require more than 250 GB of disk space to download the workloads for the full reproduction.")
        print_warning("Are you sure you want to continue? (y/n)")
        if input() != "y":
            print_error("Aborting...")
            exit(1)

    # Download AI traces
    print_info("Downloading AI traces...")
    for trace in ai_traces:
        print_info(f"Downloading {trace}...")
        src_url = AI_TRACE_URL + trace + "/nsys_reports/"
        target_dir = data_dir + "/ai/" + trace
        # Check if the directory already exists
        if os.path.exists(target_dir):
            print_warning(f"Skipping {trace} because it already exists...")
            continue
        if os.system(DOWNLOAD_CMD.format(target_dir, src_url)) != 0:
            print_error(f"Failed to download {trace}...")
            exit(1)
        print_success(f"Downloaded {trace}...")
    
    # Download HPC traces
    print_info("Downloading HPC traces...")
    for trace in hpc_traces:
        print_info(f"Downloading {trace}...")
        src_url = HPC_TRACE_URL + trace + "/mpi_traces/"
        target_dir = data_dir + "/hpc/" + trace
        # Check if the directory already exists
        if os.path.exists(target_dir):
            print_warning(f"Skipping {trace} because it already exists...")
            continue
        
        if os.system(DOWNLOAD_CMD.format(target_dir, src_url)) != 0:
            print_error(f"Failed to download {trace}...")
            exit(1)
        print_success(f"Downloaded {trace}...")
    
    # Download case studies
    print_info("Downloading case studies...")
    # TODO: TOMMASO Add download commands for case studies



def run_full_reproduction(data_dir: str) -> None:
    """
    Run a full reproduction of the artifact.

    1. Runs the validation experiment for AI workloads
    2. Runs the validation experiment for HPC workloads
    3. Runs the validation experiment for case studies
    4. Runs the case studies
    """
    print_info("Running a full reproduction...")
    # Add full reproduction commands
    download_data(data_dir, is_quick_test=False)



def run_quick_test(data_dir: str) -> None:
    """
    Run a quick functionality test of the artifact.

    The quick functionality test is a simple test that checks if the artifact is working correctly.
    It performs the following steps:
    1. Downloads the necessary data from the storage server
    2. Runs the validation experiment for AI workloads
    3. Runs the validation experimetn for HPC workloads
    4. Runs the case studies
    """
    print_info("Running a quick functionality test...")
    download_data(data_dir, is_quick_test=True)
    
    # Run the validation experiment for AI workloads
    print_info("Running the validation experiment for AI workloads...")
    # Run the validation experiment for AI workloads
    os.system(f"python {VALIDATION_SCRIPT_PATH} -d {data_dir} -t ai -q")
    # Run the validation experiment for HPC workloads
    os.system(f"python {VALIDATION_SCRIPT_PATH} -d {data_dir} -t hpc -q")

    # TODO: TOMMASO Add experiment for case studies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiments needed for reproducing the results.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("-q", "--quick", action="store_true", help="Perform a quick functionality test.")
    parser.add_argument("-f", "--full", action="store_true", help="Perform a full reproduction run.")
    parser.add_argument("-d", "--data-dir", type=str, required=True, help="Directory to store the data.")
    args = parser.parse_args()

    # Makes sure that only one of the options is provided
    if args.quick and args.full:
        print_error("Invalid option. Please use only one of -q or -f.")
        exit(1)

    if args.quick:
        print_info("Performing quick functionality test...")
        run_quick_test(args.data_dir)
    
    if args.full:
        print_info("Performing full reproduction run...")
        run_full_reproduction(args.data_dir)
        # Add full reproduction commands
    
    if not args.quick and not args.full:
        print_error("No run option provided. Please use -q or -f.")
        exit(1)
    
    print_info("Running the experiments needed for reproducing the results...")

    print_success("Done!")