import os
import argparse
import sys


def print_warning(message: str, flush: bool = True) -> None:
    print(f"\033[93m{message}\033[0m", flush=flush)

def print_error(message: str, flush: bool = True) -> None:
        print(f"\033[91m{message}\033[0m", flush=flush)

def print_success(message: str, flush: bool = True) -> None:  
    print(f"\033[92m{message}\033[0m", flush=flush)
    
def print_info(message: str, flush: bool = True) -> None:
    print(f"\033[94m{message}\033[0m", flush=flush)



def run_validation_exp(data_dir: str, app_type: str, quick_test: bool,
                       full_reproduction: bool, verbose: bool) -> None:
    """
    Runs the validation experiment for the given data directory.
    """
    print_info("Running the validation experiment...")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the validation experiment for the paper.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory containing the data.')
    parser.add_argument('-t', '--app-type', type=str, default="ai",
                        help="Type of application traces to be simulated. Options are 'hpc' and 'ai'.")
    parser.add_argument('-q', '--quick-test', action='store_true', help='Run the validation experiment for the quick test.')
    parser.add_argument('-f', '--full-reproduction', action='store_true', help='Run the validation experiment for the full reproduction.')
    args = parser.parse_args()

    # Makes sure that only one of the options is provided
    if args.quick_test and args.full_reproduction:
        print_error("Invalid option. Please use only one of -q or -f.")
        exit(1)

    if args.quick_test:
        run_validation_exp(args.data_dir, args.app_type, quick_test=True, full_reproduction=False, verbose=True)
    elif args.full_reproduction:
        run_validation_exp(args.data_dir, args.app_type, quick_test=False, full_reproduction=True, verbose=True)
    else:
        print_error("Invalid option. Please use -q or -f to specify the type of experiment to run. '-q' for the quick test and '-f' for the full reproduction.")
        exit(1)
