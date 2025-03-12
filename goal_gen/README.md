## GOAL File Generation
This directory contains the GOAL generator toolchains for AI, HPC, and storage applications.

### Merging Multiple GOAL Files to Simulate Multi-Job or Multi-Tenant Scenarios

To simulate multiple jobs or tenants on a shared network, you can merge multiple GOAL files into a single file. This is done through the `merge_goals.py` script. The placement of the jobs and other parameters can be specified through the JSON configuration file. The descriptions of the parameters in the JSON configuration file are as follows:
- "mode": The mode of merging, choices are either __"multi-job"__ or __"multi-tenant"__. In the multi-job mode, the schedules of each rank from different GOALs/applications are placed on separate nodes in the newly generated GOAL file. In the multi-tenant mode, the schedules of each rank from different GOALs/applications are placed on the same node in the newly generated GOAL file.
  - As an example, if there are two GOAL files, each with 4 ranks. Nodes are numbered from 0 to 3 from the first GOAL file and from 4 to 7 from the second GOAL file. In the multi-job mode, the ranks from the first GOAL file are placed on nodes 0 to 3, and the ranks from the second GOAL file are placed on nodes 4 to 7. In the multi-tenant mode, the ranks from the first GOAL file are placed on nodes 0 to 3, and the ranks from the second GOAL file are placed on the same nodes 0 to 3.
- "goal_files": A list of GOAL files to be merged. They do not necessarily have to have the same number of ranks.
- "pattern": The pattern of the placement of the ranks from different GOAL files. Possible options are __"round-robin"__, __"packed"__, __"random"__, or simply a list of lists of integers specifying __custom__ placement. In the custom case, each number at index i in the list corresponds to the node number where the rank i will be placed. It is very similar to how you specify the placement MPI processes with a hostfile. Note that if two ranks are placed on the same node, they will be treated the same as in the multi-tenant case. If the mode is set to 'multi-tenant', only 'packed', 'random', and 'custom' patterns are supported.
  - If we have two GOAL files with 4 ranks each. The ranks are numbered from 0 to 3 from the first GOAL file and from 4 to 7 from the second GOAL file. Examples of each of the resulting pattern options are as follows, shown in hostfile-like format:
    - "round-robin": [[0, 2, 4, 6], [1, 3, 5, 7]]
    - "packed": [[0, 1, 2, 3], [4, 5, 6, 7]]
    - "random": [[4, 0, 6, 2], [5, 1, 7, 3]]
    - Custom: [[0, 1, 2, 3], [4, 5, 6, 7]]
- "share-nic": If set to true, the ranks from different GOAL files will share the same NIC if they are placed on the same node. If set to false, each rank will have its own NIC when generating the new GOAL file. Only applicable in the multi-tenant mode.