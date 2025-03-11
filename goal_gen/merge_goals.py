import os
import argparse
import json
import random
import re


from typing import List, Dict, Optional, Tuple, Union

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


def get_config(config_path: str) -> Dict:
    """
    Load the configuration file.
    """
    if not os.path.exists(config_path):
        print_error(f"Configuration file '{config_path}' does not exist.")
        exit(1)
    with open(config_path, "r") as f:
        config = json.load(f)
    return config



def get_packed_mapping(mode: str, job_ranks: List[int],
                       verbose: bool) -> List[List[int]]:
    """
    Generates a packed mapping of ranks to nodes in the cluster.
    FIXME: Written for the sake of clarity not for brevity.
    """
    print_info("Generating packed mapping...", verbose)
    total_ranks = sum(job_ranks)
    if mode == "multi-job":
        res = []
        start = 0
        for num_ranks in job_ranks:
            res.append(list(range(start, start + num_ranks)))
            start += num_ranks
        return res
    elif mode == "multi-tenant":
        res = []
        for num_ranks in job_ranks:
            res.append(list(range(num_ranks)))
        return res
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)


def get_round_robin_mapping(mode: str, job_ranks: List[int],
                            verbose: bool) -> List[List[int]]:
    """
    Generates a round-robin mapping of ranks to nodes in the cluster.
    """
    print_info("Generating round-robin mapping...", verbose)
    if mode == "multi-job":
        # FIXME Not very efficient but it works
        num_jobs = len(job_ranks)
        total_ranks = sum(job_ranks)
        res = [[] for _ in range(num_jobs)]
        next_job = 0
        i = 0
        # Emulates the round-robin placement the hard way
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
    """
    Generates a random mapping of ranks to nodes in the cluster.
    """
    print_info("Generating random mapping...", verbose)
    if mode == "multi-job":
        total_ranks = sum(job_ranks)
        tmp = list(range(total_ranks))
        random.shuffle(tmp)
        res = []
        start = 0
        for num_ranks in job_ranks:
            res.append(tmp[start:start + num_ranks])
            start += num_ranks
        return res
    elif mode == "multi-tenant":
        max_nodes = max(job_ranks)
        res = []
        for num_ranks in job_ranks:
            res.append(random.sample(range(max_nodes), num_ranks))
        
        return res
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)


def verify_custom_pattern(mode: str, job_ranks: List[int], pattern: List[List[int]],
                          verbose: bool) -> bool:
    """
    Verifies that the custom pattern is valid.
    """
    if mode == "multi-job":
        total_ranks = sum(job_ranks)
        ranks = set()
        for i, rank_list in enumerate(pattern):
            if len(rank_list) != job_ranks[i]:
                print_warning(f"Number of ranks for job {i} does not match the expected value {job_ranks[i]}", verbose)
                return False
            for rank in rank_list:
                if rank in ranks:
                    print_warning(f"Rank {rank} is duplicated.", verbose)
                    return False
                if rank < 0 or rank >= total_ranks:
                    print_warning(f"Rank {rank} is out of range.", verbose)
                    return False
                ranks.add(rank)
        if len(ranks) != total_ranks:
            print_warning("Some ranks are missing.", verbose)
            return False
        return True
    elif mode == "multi-tenant":
        max_nodes = max(job_ranks)
        ranks = set()
        for i, rank_list in enumerate(pattern):
            if len(rank_list) != job_ranks[i]:
                print_warning(f"Number of ranks for job {i} does not match the expected value {job_ranks[i]}.", verbose)
                return False
            for rank in rank_list:
                if rank < 0 or rank >= max_nodes:
                    print_warning(f"Rank {rank} is out of range.", verbose)
                    return False
                ranks.add(rank)
        return True
    else:
        print_error(f"Invalid mode: {mode}")
        exit(1)


# ===============================================
# Main functions
# ===============================================

def load_number_of_ranks(goal_files: List[str], verbose: bool) -> List[int]:
    """
    Load the number of ranks and the list of ranks from the goal files.
    The number of ranks can be read directly from the first line
    of the goal file. It is in the format of "num_ranks <number>".
    """
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


def get_rank_mapping(mode: str, job_ranks: List[int], pattern: Union[List, str],
                     verbose: bool) -> List[List[int]]:
    """
    Generates a mapping of ranks to nodes in the cluster based on the given mode and pattern.
    Note that if the specified pattern is a string, it can be one of the following:
    - "packed"
    - "round_robin"
    - "random"
    If it is a list, then this function serves as a verification step to ensure that
    the given custom pattern is valid.
    """
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
        print_info(f"Configuration:")
        print_info(f"Mode: {config['mode']}")
        print_info(f"Goal Files: {config['goal_files']}")

    verbose = args.verbose

    job_ranks = load_number_of_ranks(config["goal_files"], verbose)

    rank_mapping = get_rank_mapping(config["mode"], job_ranks, config["pattern"], verbose)

    print_success(f"Successfully generated rank mapping: {rank_mapping}", verbose)