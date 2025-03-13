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


def rank_mapping_to_job_ranks(rank_mapping: List[List[int]]) -> Dict[int, Tuple[int, int]]:
    """
    Converts the given rank mapping to a dictionary of tuples where the tuple
    at key i contains the job index and the rank index in the job.
    Only works for multi-job mode.
    """
    res = {}
    for job, mapped_ranks in enumerate(rank_mapping):
        for i, rank in enumerate(mapped_ranks):
            res[rank] = (job, i)
    return res


def rank_remap_for_job(rank_mapping: List[List[int]]) -> List[List[int]]:
    """
    Remaps the ranks in all jobs in order to determine the sending and
    receiving ranks in the new GOAL file. Returns a list of lists where
    each list contains the new ranks for a job. Index i in the list
    corresponds to the rank in the old GOAL file, and the value at
    index i is the new rank in the new GOAL file.
    """
    res = []
    for ranks in rank_mapping:
        job_rank_remap = []
        for rank in ranks:
            job_rank_remap.append(rank)
        res.append(job_rank_remap)
    return res


def get_rank_pos_in_goal_files(goal_files: List[str], rank_mapping: List[List[int]]) \
    -> List[List[int]]:
    """
    Get the position of the start of each rank in the each of
    the goal files. Returns a list of lists where the index i of each
    list corresponds to the position of the rank i in the GOAL file.
    This is done so that we can easily extract the rank schedules
    from the GOAL files without having to traverse them multiple times.
    """
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
        assert len(rank_pos) == len(rank_mapping[i]), \
            f"Number of ranks does not match in '{goal_file}'."
        f.close()
        res.append(rank_pos)
    return res


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



def write_rank_sched_to_output(out: TextIO, rank: int, rank_remap: List[List[int]],
                               rank_pos: List[int], goal_file_path: str) -> None:
    """
    Writes the rank schedule to the output file.
    """
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
            # Try to replace the destination rank in a operation
            # with the following format as per the remapping
            # l<op_id>: send <size>b to <dst> tag <tag> cpu <cpu> nic <nic>
            remapped_dst = rank_remap[int(tokens[4])]
            prefix = " ".join(tokens[:4])
            suffix = " ".join(tokens[5:])
            out.write(f"{prefix} {remapped_dst} {suffix}\n")
        elif tokens[1] == "recv":
            assert len(tokens) >= 7, f"Invalid recv operation: {line}"
            # Try to replace the source rank in a operation
            # with the following format as per the remapping
            # l<op_id>: recv <size>b from <src> tag <tag> cpu <cpu> nic <nic>
            remapped_src = rank_remap[int(tokens[4])]
            prefix = " ".join(tokens[:4])
            suffix = " ".join(tokens[5:])
            out.write(f"{prefix} {remapped_src} {suffix}\n")
        else:
            out.write(line)


def generate_multi_job_goal(goal_files: List[str], rank_mapping: List[List[int]],
                            output_file: str, verbose: bool) -> None:
    """
    Generates a multi-job goal file based on the given rank mapping as well
    as the list of goal files.
    """
    print_info(f"Generating multi-job goal file: {output_file}...", verbose)

    job_ranks = rank_mapping_to_job_ranks(rank_mapping)

    rank_remap = rank_remap_for_job(rank_mapping)
    print("job_ranks", job_ranks)
    rank_pos = get_rank_pos_in_goal_files(goal_files, rank_mapping)
    print_info(f"Obtained the positions of ranks in the GOAL files.", verbose)

    total_ranks = sum([len(x) for x in rank_mapping])

    out = open(output_file, "w")
    # Write preamble
    out.write(f"num_ranks {total_ranks}\n\n")

    for i in tqdm(range(total_ranks), disable=not verbose):
        job, rank = job_ranks[i]
        out.write(f"rank {i} {{\n")
        write_rank_sched_to_output(out, rank, rank_remap[job], rank_pos[job], goal_files[job])
        out.write("}\n\n")

    out.close()
    print_success(f"Successfully generated multi-job goal file: {output_file}")


def generate_multi_tenant_goal(goal_files: List[str], rank_mapping: List[List[int]],
                               output_file: str, verbose: bool) -> None:
    """
    Generates a multi-tenant goal file based on the given rank mapping as well
    as the list of goal files.
    """
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
        print_info(f"Configuration:")
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