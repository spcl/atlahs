#!/bin/bash


# ---- HPC deps: defaults (override with `docker run -e VAR=...` if needed)
: "${HDF5_INSTALL_DIR:=/workspace/apps/hpc/deps/hdf5/install}"
: "${NETCDF_C_INSTALL_DIR:=/workspace/apps/hpc/deps/netcdf-c/install}"
: "${NETCDF_FORTRAN_INSTALL_DIR:=/workspace/apps/hpc/deps/netcdf-fortran/install}"

export HDF5_INSTALL_DIR NETCDF_C_INSTALL_DIR NETCDF_FORTRAN_INSTALL_DIR

# If no arguments were passed, print a usage message.
if [ "$#" -eq 0 ]; then
  echo "Usage: docker run <image> <option> [<args>]"
  echo "Options:"
  echo "  build: build the applications required for reproducing the results, options are:"
  echo "    -r: build the required applications for reproducing the results"
  echo "    -t: build all applications that are required for tracing and producing the GOAL schedules"
  echo "  run: run the benchmarks with the specified arguments, options are:"
  echo "    -q: perform a quick run to show that the artifact is functional"
  echo "    -f: perform the full run to reproduce the results"
  exit 1
fi

# Get the first argument which specifies the option
option=$1
shift


DATA_DIR="/workspace/data"

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
      "-q") 
        python3 /workspace/scripts/run.py -v -q -d $DATA_DIR
        ;;
      "-f")
        python3 /workspace/scripts/run.py -v -f -d $DATA_DIR
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

# Kill all wget processes
pkill -9 wget
# Exit the script
exit 0