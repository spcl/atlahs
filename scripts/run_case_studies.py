import os
import argparse
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, overload


OUTPUT_DIR = "/workspace/data/case_studies/"
RUN_SIMULATOR_SCRIPT_PATH = "/workspace/scripts/run_simulator.py"
LGS_CONFIG_PATH = "/workspace/scripts/configs/lgs_{}_config.yaml"
LGS_EXEC_PATH = "/workspace/sim/LogGOPSim/LogGOPSim"
HTSIM_ROOT_PATH = "/workspace/sim/htsim-backend/sim/"
HTSIM_EXEC_PATH = "/workspace/sim/htsim-backend/sim/datacenter/htsim_uec"
HTSIM_EXEC_PATH_NDP = "/workspace/sim/htsim-backend/sim/datacenter/htsim_ndp"

def print_warning(message: str, flush: bool = True) -> None:
    print(f"\033[93m[WARNING] {message}\033[0m", flush=flush)

def print_error(message: str, flush: bool = True) -> None:
    print(f"\033[91m[ERROR] {message}\033[0m", flush=flush)

def print_success(message: str, flush: bool = True) -> None:  
    print(f"\033[92m[SUCCESS] {message}\033[0m", flush=flush)
    
def print_info(message: str, flush: bool = True) -> None:
    print(f"[INFO] {message}", flush=flush)

def run_command(cmd: str, desc: str) -> None:
    print_info(f"Running {desc} command...")
    ret = os.system(cmd)
    if ret != 0:
        print_error(f"{desc} failed with exit code {ret}")
    else:
        print_success(f"{desc} completed successfully")

def run_storage(data_dir: str, overwrite: bool, verbose: bool) -> None:
    print_info(f"Running storage case study in {data_dir}...")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    case_study_3_dir = Path(OUTPUT_DIR) / "case_study_storage"
    Path(case_study_3_dir).mkdir(parents=True, exist_ok=True)

    filename1 = "1os_mprdma.tmp"
    filename2 = "1os_eqds.tmp"   
    filename3 = "8os_mprdma.tmp"
    filename4 = "8os_eqds.tmp" 

    storage_uec_no_os = (f"{HTSIM_EXEC_PATH} -lgs_flow_stats -seed 4 -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_1os.topo -goal {OUTPUT_DIR}/storage.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {case_study_3_dir}/{filename1}")
    storage_ndp_no_os = (f"{HTSIM_EXEC_PATH_NDP} -lgs_flow_stats -seed 5 -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_1os.topo -goal {OUTPUT_DIR}/storage.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 -cwnd 10000000 > {case_study_3_dir}/{filename2}")
    storage_uec_os = (f"{HTSIM_EXEC_PATH} -lgs_flow_stats -seed 4 -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_8os.topo -goal {OUTPUT_DIR}/storage.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {case_study_3_dir}/{filename3}")
    storage_ndp_os = (f"{HTSIM_EXEC_PATH_NDP} -lgs_flow_stats -seed 4 -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_8os.topo -goal {OUTPUT_DIR}/storage.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {case_study_3_dir}/{filename4}")

    # run all job allocation simulations with checks
    for name, cmd in [
        ("storage_uec_no_os", storage_uec_no_os),
        ("storage_ndp_no_os", storage_ndp_no_os),
        ("storage_uec_os", storage_uec_os),
        ("storage_ndp_os", storage_ndp_os),
    ]:
        run_command(cmd, name)

def run_job_alloc(data_dir: str, overwrite: bool, verbose: bool) -> None:
    print_info(f"Running Job Allocation case study in {data_dir}...")
    # ensure the main output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    case_study_3_dir = Path(OUTPUT_DIR) / "case_study_job_alloc"
    Path(case_study_3_dir).mkdir(parents=True, exist_ok=True)

    llama_packed = f"{HTSIM_EXEC_PATH} -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_8os.topo -goal {OUTPUT_DIR}/llama_packed.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > {case_study_3_dir}/llama_packed.tmp"
    llama_random = f"{HTSIM_EXEC_PATH} -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_8os.topo -goal {OUTPUT_DIR}/llama_random.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > {case_study_3_dir}/llama_random.tmp"
    lulesh_packed = f"{HTSIM_EXEC_PATH} -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_8os.topo -goal {OUTPUT_DIR}/lulesh_packed.bin  -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > {case_study_3_dir}/lulesh_packed.tmp"
    lulesh_random = f"{HTSIM_EXEC_PATH} -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_128_8os.topo -goal {OUTPUT_DIR}/lulesh_random.bin -linkspeed 200000 -nodes 128 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > {case_study_3_dir}/lulesh_random.tmp"

    # run all job allocation simulations with checks
    for name, cmd in [
        ("llama_packed", llama_packed),
        ("llama_random", llama_random),
        ("lulesh_packed", lulesh_packed),
        ("lulesh_random", lulesh_random),
    ]:
        run_command(cmd, name)

def run_lgs_vs_htsim(data_dir: str, overwrite: bool, verbose: bool) -> None:
    print_info(f"Running LGS vs HTSIM case study in {data_dir}...")
    # ensure the main output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    case_study_2_dir = Path(OUTPUT_DIR) / "case_study_lgs_vs_htsim"
    Path(case_study_2_dir).mkdir(parents=True, exist_ok=True)

    lgs_cmd = f"{LGS_EXEC_PATH} -f {OUTPUT_DIR}/llama_lgs_vs_htsim.bin -L 3700 -o 200 -g 5 -G 0.04  > {case_study_2_dir}/llama_lgs.tmp"
    llama_no_os_htsim = f"{HTSIM_EXEC_PATH} -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_32_1os.topo -goal {OUTPUT_DIR}/llama_lgs_vs_htsim.bin -linkspeed 200000 -nodes 32 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > {case_study_2_dir}/llama_no_os_htsim.tmp"
    llama_os_htsim = f"{HTSIM_EXEC_PATH} -topo {HTSIM_ROOT_PATH}/datacenter/topologies/leaf_spine_32_4os.topo -goal {OUTPUT_DIR}/llama_lgs_vs_htsim.bin  -linkspeed 200000 -nodes 32 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > {case_study_2_dir}/llama_os_htsim.tmp"

    # run all job allocation simulations with checks
    for name, cmd in [
        ("llama_no_os_htsim", llama_no_os_htsim),
        ("llama_os_htsim", llama_os_htsim),
        ("llama_lgs", lgs_cmd),
    ]:
        run_command(cmd, name)

def run_case_studies(data_dir: str, overwrite: bool, verbose: bool) -> None:
    print_info(f"Running case studies in {data_dir}...")
    case_studies_path = Path(data_dir) / "case_studies"
    current_case_studies = ["llama_lgs_vs_htsim", "job_alloc", "storage"]
    for type_run in current_case_studies:
        if "llama_lgs_vs_htsim" in type_run:
            continue
            run_lgs_vs_htsim(data_dir, overwrite, verbose)
        elif "job_alloc" in type_run:
            continue
            run_job_alloc(data_dir, overwrite, verbose)
        elif "storage" in type_run:
            run_storage(data_dir, overwrite, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the validation experiment for the paper.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory containing the data.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the existing results.')

    args = parser.parse_args()

    # ensure output directory root exists before any case study
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    run_case_studies(args.data_dir, args.overwrite, verbose=True)
