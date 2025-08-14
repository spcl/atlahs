## AI Applications

This directory contains a collection of AI applications.

We provide Dockerfiles (`dockerfiles/`) that can be used to build the Docker image that contains the dependencies for the AI applications. The Dockerfile `Dockerfile.common` is a base image that supports both DLRM, Megatron, and MMaDA. The Dockerfile `Dockerfile.vllm` is a base image that supports vLLM (v0.8.6).

In addition, as a reference, we also provide scripts (`scripts/`) that can be used to run and trace each of the applications. Note that the scripts specifically target the Alps supercomputer, and may need to be adapted to other systems.

Before running the scripts, make sure to apply the patches to the source code of the applications to incorporate the necessary instrumentation code. The patches are provided in the `patches/` directory. Technically, GOAL files can still be successfully generated even without the patches, but collected traces may include sections of code that are not as relevant to the network-intensive and performance-critical parts of the applications.