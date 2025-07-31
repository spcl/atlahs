## ATLAHS: An Application-centric Network Simulator Toolchain for AI, HPC, and Storage

![Overview](docs/overview.png)

This repository contains the source code for ATLAHS, a network simulator toolchain for AI, HPC, and storage applications. It contains the following components:
- GOAL (Group Operation Assembly Language) generators that traces AI, HPC, and storage applications and converts them into network workloads usable by network simulators
- Various backends for simulating network workloads, including LogGOPSim, HTSim, and NS-3

The paper of this work is available on arXiv: [https://arxiv.org/pdf/2505.08936](https://arxiv.org/pdf/2505.08936).

### Docker Environment
To facilitate the reproducibility of the results which we publish in the paper, we provide a Docker image that contains all the dependencies that are required to run the ATLAHS toolchain.

To build the Docker image, run the following command:

```bash
docker build -t atlahs .
```