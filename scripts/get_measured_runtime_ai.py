import os
import sys
import json
import sqlite3
from typing import Dict, Optional, List


def get_ranks(result_dir: str) -> Dict[str, int]:
    """
    Retrieves the ranks of each compute node from the JSON file
    `nsys_events_intermediate_output.json` in the result directory.
    """
    assert os.path.exists(result_dir), f"Directory {result_dir} does not exist"
    intermediate_output = os.path.join(result_dir, "nsys_events_intermediate_output.json")
    assert os.path.exists(intermediate_output), f"File {intermediate_output} does not exist"

    with open(intermediate_output) as f:
        data = json.load(f)

    assert "hostname_to_rank" in data, "[ERROR] Wrong format: Key 'hostname_to_rank' not found in JSON file"
    return data["hostname_to_rank"]


def nsys_profiling_marks_present(result_dir: str, host_to_rank: Dict[str, int]) -> Optional[float]:
    """
    Checks if the nsys profiling marks are present in the intermediate output file.
    If they are present, returns the actual runtime of the application.
    None otherwise.
    """
    res = None
    for hostname, rank in host_to_rank.items():
        db_file = os.path.join(result_dir, "nsys_reports", f"nsys_report_{hostname}.sqlite")
        assert os.path.exists(db_file), f"File {db_file} does not exist"

        # Connects to the SQLite database file
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        profile_mark_query = "SELECT text, start FROM NVTX_EVENTS WHERE text LIKE 'nsys profiling%'"
        cursor.execute(profile_mark_query)
        pid_start = {}
        for row in cursor.fetchall():
            assert len(row) == 2, f"Expected 1 column, but got {len(row)} columns"
            text, start = row
            assert text.startswith("nsys profiling"), f"Expected text to start with 'nsys profiling', but got {text}"
            assert "pid" in text, f"Expected 'pid' to be in text, but got {text}"
            tokens = text.split(":")
            pid = int(tokens[1].strip())
            if text.startswith("nsys profiling start"):
                pid_start[pid] = start
            elif text.startswith("nsys profiling stopped"):
                assert pid in pid_start, f"pid {pid} not found in pid_start"
                runtime = start - pid_start[pid]
                res = max(res, runtime) if res is not None else runtime
                del pid_start[pid]
            else:
                raise ValueError(f"Unexpected text: {text}")
        assert len(pid_start) == 0, f"Found {len(pid_start)} unpaired pid(s) in pid_start"
    
    return res


def get_actual_runtime(result_dir: str, host_to_rank: Dict[str, int]) -> None:
    """
    Retrieves the actual runtime of each rank from the SQLite database files
    """
    for hostname, rank in host_to_rank.items():
        db_file = os.path.join(result_dir, "nsys_reports", f"nsys_report_{hostname}.sqlite")
        assert os.path.exists(db_file), f"File {db_file} does not exist"

        # Connects to the SQLite database file
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        start_time_query = "SELECT kern.start, sid_dn.value AS demangledNameStr FROM CUPTI_ACTIVITY_KIND_KERNEL AS kern JOIN StringIds AS sid_dn ON sid_dn.id = kern.demangledName WHERE demangledNameStr LIKE 'ncclDevKernel%' ORDER BY start ASC LIMIT 1"
        cursor.execute(start_time_query)
        for row in cursor.fetchall():
            assert len(row) == 2, f"Expected 2 column, but got {len(row)} columns"
            start_time, _ = row
        
        end_time_query = "SELECT kern.end, sid_dn.value AS demangledNameStr FROM CUPTI_ACTIVITY_KIND_KERNEL AS kern JOIN StringIds AS sid_dn ON sid_dn.id = kern.demangledName WHERE demangledNameStr LIKE 'ncclDevKernel%' ORDER BY end DESC LIMIT 1"
        cursor.execute(end_time_query)
        for row in cursor.fetchall():
            assert len(row) == 2, f"Expected 2 column, but got {len(row)} columns"
            end_time, _ = row
        
        actual_runtime = end_time - start_time

        print(f"Rank {rank} on {hostname} took {actual_runtime} ns ({actual_runtime / 1e9} s)")

    return actual_runtime

if __name__ == "__main__":
    # Parses the first argument as the directory which contains the SQLite database files
    # as well as the intermediate output files
    if len(sys.argv) == 1:
        result_dir = "results"
        print(f"Using default result directory: {result_dir}")
    else:
        result_dir = sys.argv[1]
        print(f"Using result directory: {result_dir}")

    # Retrieves the ranks of each compute node from the JSON file
    ranks = get_ranks(result_dir)

    # Retrieves the actual runtime of each rank from the SQLite database files
    profiling_mark_runtime = nsys_profiling_marks_present(result_dir, ranks)
    if profiling_mark_runtime is None:
        get_actual_runtime(result_dir, ranks)
    else:
        print(f"Found nsys profiling markers in the intermediate output file")
        print(f"Measured runtime: {profiling_mark_runtime} ns ({profiling_mark_runtime / 1e9} s)")
    