## ATLAHS Simulator Backends

This directory contains the simulator backends for ATLAHS, which currently includes:
- `LogGOPSim`: flow-level / message-level simulator based on the LogGOPS model
- `htsim`: packet-level simulator for more accurate and complex simulation of the workloads

For more details, please refer to the README files in the respective directories. In addition, we are planning to include more simulator backends such as `ns-3` in the future. If you are interested in integrating the front-end GOAL scheduler into your own simulator, you can read the [ATLAHS paper](https://arxiv.org/pdf/2505.08936) regarding which interfaces need to be implemented.