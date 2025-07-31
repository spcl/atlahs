#!/bin/bash

# If no arguments were passed, print a usage message.
if [ "$#" -eq 0 ]; then
  echo "Usage: docker run <image> <option> [<args>]"
  echo "Options:"
  echo "  build: build the applications required for reproducing the results, options are:"
  echo "    -r: build the required applications for reproducing the results"
  echo "    -t: build all applications that are required for tracing and producing the GOAL schedules"
  echo "  run: run the benchmarks with the specified arguments, options are:"
  echo "    -v: run the validation experiments for the artifact"
  echo "    -q: perform a quick run to show that the artifact is functional"
  echo "    -f: perform the full run to reproduce the results"
  exit 1
fi

# Get the first argument which specifies the option
option=$1
shift

case $option in
  "build")
    if [ "$#" -eq 0 ]; then
      echo "Error: Build option requires additional arguments (-r or -t)"
      exit 1
    fi

    case $1 in
        "-r")
            echo "Building required applications for reproducing the results..."
            # Add build commands here
            python3 /workspace/scripts/build.py -r -v
            ;;
        "-t")
            echo "Building all applications that are required for tracing and producing the GOAL schedules..."
            # Add build commands here
            python3 /workspace/scripts/build.py -t -v
            ;;
        *)
            echo "Error: Invalid build argument. Must be -r or -t"
            exit 1
            ;;
    esac
    ;;
    
  "run")
    if [ "$#" -eq 0 ]; then
      echo "Error: Run option requires additional arguments (-v, -q, or -f)"
      exit 1
    fi
    
    case $1 in
      "-v")
        echo "Running validation experiments..."
        # Add validation experiment commands
        ;;
      "-q") 
        echo "Performing quick functionality test..."
        # Add quick test commands
        ;;
      "-f")
        echo "Performing full reproduction run..."
        # Add full reproduction commands
        ;;
      *)
        echo "Error: Invalid run argument. Must be -v, -q, or -f"
        exit 1
        ;;
    esac
    ;;
    
  *)
    echo "Error: Invalid option. Must be 'build' or 'run'"
    exit 1
    ;;
esac
