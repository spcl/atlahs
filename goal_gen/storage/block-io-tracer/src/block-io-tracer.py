# This script will be used to trace block I/O operations on a Linux system.
# It will use the blktrace utility to trace block I/O operations on specific devices.

import os
import sys
import subprocess
import argparse
import signal
import time
from typing import List, Optional, Dict

trace_started = False

def print_warning(msg: str):
    print('\033[93m' + "[DEBUG] " + msg + '\033[0m')

def print_error(msg: str):
    print('\033[91m' + "[DEBUG] " + msg + '\033[0m')

def write_bpf_script(apps: List[str], time: Optional[int] = None) -> None:
    """
    Creates a BPF script to trace block I/O operations for the specified application.
    :param apps: List of application names
    :return: None
    """
    app_ids = {app: i for i, app in enumerate(apps)}

    script = f'''#!/usr/bin/env bpftrace
// This script traces block I/O operations for the specified applications
// Usage: bpftrace io-trace.bt

'''
    for app, app_id in app_ids.items():
        script += f'''tracepoint:block:block_rq_issue
/comm == "{app}"/
{{
  printf("%d,%llu,%u,%s,%llu\\n", {app_ids[app]}, args->sector, args->bytes, args->rwbs, elapsed);
}}'''

    if time:
        print(f"Tracing for {time} seconds")
        script += f'''
interval:s:1
{{
  // Check how many seconds have elapsed
  if (elapsed > {time} * 1000000000)
  {{
    // After 10 seconds, exit bpftrace gracefully
    printf("Time limit reached. Exiting...\\n");
    exit();
  }}
}}'''

    with open('io-trace.bt', 'w') as f:
        f.write(script)
    
    print(f"Generated BPF script: io-trace.bt")


def trace_io_operations(output_file: str) -> None:
    """
    Traces block I/O operations on the system using the generated BPF script.
    :param output_file: Name of the output file
    :param time: Time to trace in seconds
    :param silent: Do not print the trace output to the console
    :return: None
    """
    cmd = ['bpftrace', 'io-trace.bt', '-o', output_file]
    print(f"Tracing block I/O operations for {', '.join(args.apps)}...")
    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        while True:
            time.sleep(1)
            if proc.returncode is not None:
                break
    except KeyboardInterrupt:
        print("Exiting...")
        if proc.returncode is None:
            proc.send_signal(signal.SIGINT)
        proc.wait()
        print("Tracing stopped")
    
    if proc.returncode != 0:
        print_error(f"Error occurred while tracing block I/O operations: {proc.stderr}")
        sys.exit(1)
    
    print(f"Block I/O trace saved to '{output_file}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Block I/O Tracer')
    parser.add_argument('-a', '--apps', help='Name of the applications to trace', required=True, nargs='+')
    parser.add_argument('-o', '--output', help='Name of the output file, default is <hostname>-io-trace.io', default=None)
    parser.add_argument('-t', '--time', help='Time to trace in seconds, if not specified, it will trace indefinitely',
                        type=int)
    args = parser.parse_args()
    
    print_warning("You need 'sudo' to run this script")
    # Checks if the blktrace utility is installed on the system
    try:
        res = subprocess.run(['bpftrace', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert res.returncode == 0
    except:
        print_error("'bpftrace' is not installed on the system")
        sys.exit(1)
    
    if args.output:
        output_file_path = args.output
    else:
        output_file_path = f'{os.uname().nodename}-io-trace.io'
    
    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print_warning(f"Output file '{output_file_path}' already exists. It will be overwritten.")
    
    # Check if the specified applications are running
    for app in args.apps:
        res = subprocess.run(['pgrep', app], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print_error(f"Application '{app}' is not running")
            sys.exit(1)
    
    write_bpf_script(args.apps, args.time)
    
    trace_io_operations(output_file_path)
    
    
