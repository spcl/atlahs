import os
import re
import sys
import argparse
import subprocess
import time

CONVERION_SCRIPT = "nsys_reports_to_sqlite.sh"
GET_RUNTIME_SCRIPT = "get_measured_runtime_ai.py"
SLURM_JOB_CHECK_INTERVAL = 30

# =================================================
# Utility functions
# =================================================

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


def fetch_slurm_job_id(output: str) -> str:
    """
    Fetches the SLURM job ID from the output.
    """
    match = re.search(r"Submitted batch job (\d+)", output)
    if match:
        job_id = match.group(1)
        return job_id
    else:
        print_error(f"Failed to fetch the job ID from the output: {output}")
        exit(1)


def slurm_job_finished(job_id: str, check_interval: int = 10) -> bool:
    """
    Checks if the SLURM job has finished.
    """
    cmd = ["sacct", "-j", job_id, "--noheader", "--format=state"]

    time.sleep(check_interval)
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print_error(f"Failed to run the command: {result.stderr.decode()}")
        exit(1)
    
    states = [line.strip() for line in result.stdout.decode().splitlines() if line.strip()]
    if len(states) == 0:
        print_error("Failed to fetch the job state")
        exit(1)
    
    return states[0] == "COMPLETED" or states[0] == "CANCELLED" or states[0] == "FAILED"

# =================================================
# Main function
# =================================================


def run_ai_app(output_file: str, num_trials: int, command: str, trace_dir: str,
               verbose: bool = True) -> None:
    """
    Runs the AI application `num_trials` times and saves the runtime of each trial in the output file.
    """
    # Appends the measured runtime of the AI application to the output file
    output = open(output_file, "a")
    print_info(f"Running the AI application {num_trials} times...", verbose)
    print_info(f"Command: {command}", verbose)
    cmd = command.strip().split()
    assert len(cmd) > 0, "Command is empty"

    is_slurm = cmd[0] == "sbatch"
    print_info(f"SLURM job detected", verbose)

    for i in range(num_trials):
        print_info(f"Running trial {i+1}/{num_trials}...")

        # Clean the trace directory
        if os.path.exists(trace_dir):
            print_info("Cleaning the trace directory...", verbose)
            clean_cmd = ["rm", "-rf", trace_dir]
            result = subprocess.run(clean_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print_error(f"Failed to run the command: {result.stderr.decode()}", verbose)
                exit(1)
        os.makedirs(trace_dir, exist_ok=True)
        print_info("Trace directory cleaned", verbose)
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print_error(f"Failed to run the command: {result.stderr.decode()}", verbose)
            exit(1)
        
        if is_slurm:
            job_id = fetch_slurm_job_id(result.stdout.decode())
            print_info(f"SLURM Job ID: {job_id}", verbose)

            # Waits for the SLURM job to finish
            # Note: This is a blocking call
            print_info("Waiting for the SLURM job to finish...", verbose)
            while not slurm_job_finished(job_id, SLURM_JOB_CHECK_INTERVAL):
                print_info(f"SLURM job {job_id} is still running...", verbose)
            print_success(f"SLURM job {job_id} has finished", verbose)
        

        print_info(f"Checking the trace directory {trace_dir}...", verbose)
        if not os.path.exists(trace_dir):
            print_error(f"Trace directory {trace_dir} does not exist", verbose)
            exit(1)
        
        # Get the number of nsys-report files in the trace directory
        nsys_report_files = [f for f in os.listdir(trace_dir) if f.endswith("nsys-rep")]
        if len(nsys_report_files) == 0:
            print_error("No nsys-report files found in the trace directory", verbose)
            exit(1)
        
        print_info(f"Number of nsys-report files: {len(nsys_report_files)}", verbose)
        num_nodes = len(nsys_report_files)

        # Converts the nsys-report files in the trace directory to SQLite database files
        print_info("Converting nsys-report files to SQLite database files...", verbose)
        convert_cmd = ["bash", CONVERION_SCRIPT, trace_dir]
        
        result = subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print_error(f"Failed to run the command: {result.stderr.decode()}", verbose)
            exit(1)
        
        assert len([f for f in os.listdir(trace_dir) if f.endswith("sqlite")]) == num_nodes, "Number of SQLite database files does not match the number of nodes"
        print_success("Converted nsys-report files to SQLite database files", verbose)
        
        # Retrieves the measured runtime of the AI application from the traces
        print_info("Retrieving the measured runtime of the AI application...", verbose)
        get_runtime_cmd = ["python3", GET_RUNTIME_SCRIPT, trace_dir]
        result = subprocess.run(get_runtime_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print_error(f"Failed to run the command: {result.stderr.decode()}", verbose)
            exit(1)
        
        process_out = result.stdout.decode().strip()
        # Output "Measured runtime: {profiling_mark_runtime} ns"
        pattern = re.compile(r"Measured runtime: (\d+) ns")
        match = pattern.search(process_out)
        if match:
            runtime = int(match.group(1))
            output.write(f"{i+1},{runtime}\n")
            output.flush()
            print_success(f"Measured runtime: {runtime} ns ({runtime / 1e9} s) for trial {i + 1}", verbose)
        else:
            print_error(f"Failed to parse the output: {output}", verbose)
            exit(1)

    output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI application and measure its runtime")
    parser.add_argument("-o", "--output-file", type=str,
                        default="ai_app.out", help="Output file of the results in CSV format")
    parser.add_argument('-n', "--num-trials", type=int,
                        default=10, help="Number of trials to run the AI application")
    parser.add_argument("-c", "--command", type=str, required=True,
                        help="Command to run the application and save the trace")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the results stored output file if it exists")
    parser.add_argument("--trace-dir", required=True, type=str,
                        help="Directory to save the traces")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Retrieves the measured runtime of the AI application from the traces
    output_file = args.output_file
    if args.overwrite and os.path.exists(output_file):
        os.remove(output_file)
        with open(output_file, "w") as output:
            output.write("trial,runtime\n")

    # Get the absolute path of the trace dir
    trace_dir = os.path.abspath(args.trace_dir)

    print_info(f"Output file: {output_file}")
    print_info(f"Number of trials: {args.num_trials}")
    print_info(f"Command: {args.command}")
    print_info(f"Trace directory: {trace_dir}")

    run_ai_app(output_file, args.num_trials, args.command, trace_dir, args.verbose)