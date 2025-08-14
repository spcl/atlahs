## HPC Applications

This directory contains a collection of HPC applications.

The Dockerfile in this directory is a base image that contains the dependencies for all the HPC applications.

The `build_apps.py` script can be used to build the applications. Executing the following command in the root directory in the Docker container will also build all of the applications:
```bash
> docker run -v $(pwd):/workspace atlahs build -t
```