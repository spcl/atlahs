import os
import argparse
import json
import re


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



def 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output GOAL file")
    args = parser.parse_args()
    
