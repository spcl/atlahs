import os
import argparse
import json
import random
import re
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union, TextIO

# ===============================================
# Utility functions
# ===============================================

def print_warning(message: str, verbose: bool = True, flush: bool = True) -> None:
    if verbose:
        CSTART = '\033[93m'
        CEND = '\033[0m'
        print(f"{CSTART}[WARNING] {message}{CEND}", flush=flush)

def print_error(message: str, verbose: bool = True, flush: bool = True) -> None:
    if verbose:
        CSTART = '\033[91m'
        CEND = '\033[0m'
        print(f"{CSTART}[ERROR] {message}{CEND}", flush=flush)

def print_success(message: str, verbose: bool = True, flush: bool = True) -> None:
    if verbose:
        CSTART = '\033[92m'
        CEND = '\033[0m'
        print(f"{CSTART}[SUCCESS] {message}{CEND}", flush=flush)

def print_info(message: str, verbose: bool = True, flush: bool = True) -> None:
    if verbose:
        print(f"[INFO] {message}", flush=flush)

def get_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        print_error(f"Configuration file '{config_path}' does not exist.")
        exit(1)
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

# ===============================================
# Mapping functions
# ===============================================

def get_packed_mapping(mode: str, job_ranks: List[int], verbose: bool) -> List[List[int]]:
    print_info("Generating packed mapping...", verbose)
    total_ranks = sum(job_ranks)
    if mode == "multi-job":
        res = []
        start = 0
        for num in job_ranks:
            res.append(list(range(start, start + num)))
            start += num
        return res
    elif mode == "multi-tenant":
        res = []
        for num in job_ranks:
            res.append(list(range(num)))
        return res
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)

def get_round_robin_mapping(mode: str, job_ranks: List[int], verbose: bool) -> List[List[int]]:
    print_info("Generating round-robin mapping...", verbose)
    if mode == "multi-job":
        num_jobs = len(job_ranks)
        total_ranks = sum(job_ranks)
        res = [[] for _ in range(num_jobs)]
        next_job = 0
        i = 0
        while i < total_ranks:
            if len(res[next_job]) < job_ranks[next_job]:
                res[next_job].append(i)
                i += 1
            next_job = (next_job + 1) % num_jobs
        assert max([max(x) for x in res]) == total_ranks - 1, f"Invalid round-robin mapping: {res}"
        assert sum([len(x) for x in res]) == total_ranks, f"Invalid round-robin mapping: {res}"
        return res
    elif mode == "multi-tenant":
        print_error("Round-robin mapping is not supported for multi-tenant mode.")
        exit(1)
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)

def get_random_mapping(mode: str, job_ranks: List[int], verbose: bool) -> List[List[int]]:
    print_info("Generating random mapping...", verbose)
    if mode == "multi-job":
        total_ranks = sum(job_ranks)
        tmp = list(range(total_ranks))
        random.shuffle(tmp)
        res = []
        start = 0
        for num in job_ranks:
            res.append(tmp[start:start + num])
            start += num
        return res
    elif mode == "multi-tenant":
        max_nodes = max(job_ranks)
        res = []
        for num in job_ranks:
            res.append(random.sample(range(max_nodes), num))
        return res
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)

def verify_custom_pattern(mode: str, job_ranks: List[int], pattern: List[List[int]], verbose: bool) -> bool:
    if mode == "multi-job":
        # In custom mode we expect each inner list length to match the expected number of ranks.
        for i, rank_list in enumerate(pattern):
            if len(rank_list) != job_ranks[i]:
                print_warning(f"Job {i} pattern length {len(rank_list)} does not match expected {job_ranks[i]}", verbose)
                return False
            for rank in rank_list:
                if rank < 0:
                    print_warning(f"Rank {rank} is negative in job {i}", verbose)
                    return False
        return True
    elif mode == "multi-tenant":
        max_nodes = max(job_ranks)
        for i, rank_list in enumerate(pattern):
            if len(rank_list) != job_ranks[i]:
                print_warning(f"Job {i} pattern length {len(rank_list)} does not match expected {job_ranks[i]}", verbose)
                return False
            for rank in rank_list:
                if rank < 0 or rank >= max_nodes:
                    print_warning(f"Rank {rank} is out of range in job {i}", verbose)
                    return False
        return True
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)

def rank_mapping_to_job_ranks(rank_mapping: List[List[int]]) -> Dict[int, Tuple[int,int]]:
    """
    Converts the custom mapping to a dictionary mapping global rank -> (job index, rank index in that job).
    For example, with pattern [[0,3,20,30], [4,5,6,7,8,9,10,11]] the dictionary will contain:
      0 -> (0,0), 3 -> (0,1), 20 -> (0,2), 30 -> (0,3)
      4 -> (1,0), 5 -> (1,1), etc.
    """
    res = {}
    for job, mapped in enumerate(rank_mapping):
        for idx, global_rank in enumerate(mapped):
            if global_rank in res:
                print_warning(f"Global rank {global_rank} is assigned more than once.", True)
                exit(1)
            res[global_rank] = (job, idx)
    return res

def rank_remap_for_job(rank_mapping: List[List[int]]) -> List[List[int]]:
    """
    For custom mapping we simply return the mapping as provided.
    """
    return rank_mapping

# ===============================================
# GOAL file processing functions
# ===============================================

def get_rank_pos_in_goal_files(goal_files: List[str], rank_mapping: List[List[int]]) -> List[List[int]]:
    res = []
    for i, goal_file in enumerate(goal_files):
        f = open(goal_file, "r")
        rank_pos = []
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("rank"):
                rank_pos.append(f.tell())
        # Expect number of rank blocks equal to the pattern count for that job.
        assert len(rank_pos) == len(rank_mapping[i]), f"Number of ranks does not match in '{goal_file}'."
        f.close()
        res.append(rank_pos)
    return res

def load_number_of_ranks(goal_files: List[str], verbose: bool) -> List[int]:
    ranks = []
    for goal_file in goal_files:
        if not os.path.exists(goal_file):
            print_error(f"Goal file '{goal_file}' does not exist.")
            exit(1)
        with open(goal_file, "r") as f:
            num_ranks = None
            for line in f:
                if line.startswith("num_ranks"):
                    num_ranks = int(line.split()[1])
                    break
            if num_ranks is None:
                print_error(f"Number of ranks not found in '{goal_file}'.")
                exit(1)
            ranks.append(num_ranks)
            print_info(f"Number of ranks in '{goal_file}': {num_ranks}", verbose)
    return ranks

def get_rank_mapping(mode: str, job_ranks: List[int], pattern: Union[List, str], verbose: bool) -> List[List[int]]:
    assert mode in ("multi-job", "multi-tenant"), f"Invalid mode: {mode}"
    if isinstance(pattern, str):
        if pattern == "packed":
            return get_packed_mapping(mode, job_ranks, verbose)
        elif pattern == "round_robin":
            return get_round_robin_mapping(mode, job_ranks, verbose)
        elif pattern == "random":
            return get_random_mapping(mode, job_ranks, verbose)
        else:
            print_error(f"Invalid pattern: {pattern}")
            exit(1)
    elif isinstance(pattern, list):
        if verify_custom_pattern(mode, job_ranks, pattern, verbose):
            print_success("Custom pattern is verified.", verbose)
            return pattern
        else:
            print_error(f"Invalid custom pattern: {pattern}")
            exit(1)
    else:
        print_error(f"Invalid pattern: {pattern}")
        exit(1)

def write_rank_sched_to_output(out: TextIO, rank: int, rank_remap: List[int], rank_pos: List[int], goal_file_path: str) -> None:
    goal_file = open(goal_file_path, "r")
    goal_file.seek(rank_pos[rank])
    while True:
        line = goal_file.readline()
        if not line or line.startswith("}"):
            break
        tokens = line.split()
        if len(tokens) == 0:
            continue
        if tokens[1] == "send":
            assert len(tokens) >= 7, f"Invalid send operation: {line}"
            remapped_dst = rank_remap[int(tokens[4])]
            prefix = " ".join(tokens[:4])
            suffix = " ".join(tokens[5:])
            out.write(f"{prefix} {remapped_dst} {suffix}\n")
        elif tokens[1] == "recv":
            assert len(tokens) >= 7, f"Invalid recv operation: {line}"
            remapped_src = rank_remap[int(tokens[4])]
            prefix = " ".join(tokens[:4])
            suffix = " ".join(tokens[5:])
            out.write(f"{prefix} {remapped_src} {suffix}\n")
        else:
            out.write(line)
    goal_file.close()

# ===============================================
# Main functions
# ===============================================

def generate_multi_job_goal(goal_files: List[str], rank_mapping: List[List[int]], output_file: str, verbose: bool) -> None:
    print_info(f"Generating multi-job goal file: {output_file}...", verbose)
    # Build dictionary: global rank -> (job index, rank index in that job)
    global_map = rank_mapping_to_job_ranks(rank_mapping)
    # Total global ranks is max(global rank) + 1
    total_ranks = max(global_map.keys()) + 1
    # Obtain remapping from custom pattern (remains unchanged)
    remap_by_job = rank_remap_for_job(rank_mapping)
    # Obtain positions in each GOAL file based on expected number of ranks (pattern length)
    rank_pos_by_job = get_rank_pos_in_goal_files(goal_files, rank_mapping)

    out = open(output_file, "w")
    out.write(f"num_ranks {total_ranks}\n\n")

    # For each global rank, if it was mapped then use that job's schedule; otherwise create an empty block.
    for global_rank in tqdm(range(total_ranks), disable=not verbose):
        out.write(f"rank {global_rank} {{\n")
        if global_rank in global_map:
            job, job_rank = global_map[global_rank]
            write_rank_sched_to_output(out, job_rank, remap_by_job[job], rank_pos_by_job[job], goal_files[job])
        # If no rank mapping is provided for this global rank, leave the block empty.
        out.write("}\n\n")
    out.close()
    print_success(f"Successfully generated multi-job goal file: {output_file}", verbose)

def generate_multi_tenant_goal(goal_files: List[str], rank_mapping: List[List[int]], output_file: str, verbose: bool) -> None:
    raise NotImplementedError("Multi-tenant goal generation is not implemented yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output GOAL file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose messages")
    args = parser.parse_args()

    config = get_config(args.config)
    assert "mode" in config, "'mode' not found in the configuration file."
    assert "goal_files" in config, "'goal_files' not found in the configuration file."
    assert "pattern" in config, "'pattern' not found in the configuration file."
    
    if args.verbose:
        print_info("Configuration:")
        print_info(f"Mode: {config['mode']}")
        print_info(f"Goal Files: {config['goal_files']}")

    verbose = args.verbose
    job_ranks = load_number_of_ranks(config["goal_files"], verbose)
    rank_mapping = get_rank_mapping(config["mode"], job_ranks, config["pattern"], verbose)
    print_success(f"Successfully generated rank mapping: {rank_mapping}", verbose)

    if config["mode"] == "multi-job":
        generate_multi_job_goal(config["goal_files"], rank_mapping, args.output, verbose)
    elif config["mode"] == "multi-tenant":
        generate_multi_tenant_goal(config["goal_files"], rank_mapping, args.output, verbose)
    else:
        print_error(f"Invalid mode: {config['mode']}")
        exit(1)