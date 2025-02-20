#!/bin/bash

podman build -t hpc_apps .
export XDG_RUNTIME_DIR=/dev/shm/$USER/xdg_runtime_dir
mkdir -p $XDG_RUNTIME_DIR

enroot import -o hpc_apps_new.sqsh podman://hpc_apps

# remove sqsh if exists
rm -f hpc_apps.sqsh

mv hpc_apps_new.sqsh hpc_apps.sqsh
