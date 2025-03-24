import yaml

def get_nccl_btree(nranks, rank):
    up, down0, down1 = -1, -1, -1
    parent_child_type = -1

    # Find the highest set bit in the rank
    bit = 1
    while bit < nranks:
        if bit & rank:
            break
        bit <<= 1

    if rank == 0:
        # Root node has no parent or left child
        up = -1
        down0 = -1
        # Right child is the first node in the second tree
        down1 = bit >> 1 if nranks > 1 else -1
        return {
            "u": up,
            "d0": down0,
            "d1": down1,
            "parent_child_type": -1  # Root has no parent
        }

    # Calculate parent (up)
    up = (rank ^ bit) | (bit << 1)
    if up >= nranks:
        up = rank ^ bit

    # Determine parent-child relationship type
    parent_child_type = 0 if rank < up else 1

    # Calculate children (down0 and down1)
    lowbit = bit >> 1
    down0 = rank - lowbit if lowbit != 0 else -1
    down1 = rank + lowbit if lowbit != 0 else -1

    # Ensure down1 is within bounds
    while down1 >= nranks:
        lowbit >>= 1
        down1 = rank + lowbit if lowbit != 0 else -1

    return {
        "u": up,
        "d0": down0,
        "d1": down1,
        "parent_child_type": parent_child_type
    }


def get_nccl_dtree(nranks, rank):
    # First tree (binary tree)
    tree0 = get_nccl_btree(nranks, rank)
    s0 = tree0["u"]
    d0_0 = tree0["d0"]
    d0_1 = tree0["d1"]
    parent_child_type0 = tree0["parent_child_type"]

    # Second tree (mirror or shift)
    if nranks % 2 == 1:
        # Shift logic
        shiftrank = (rank - 1 + nranks) % nranks
        tree1 = get_nccl_btree(nranks, shiftrank)
        s1 = -1 if tree1["u"] == -1 else (tree1["u"] + 1) % nranks
        d1_0 = -1 if tree1["d0"] == -1 else (tree1["d0"] + 1) % nranks
        d1_1 = -1 if tree1["d1"] == -1 else (tree1["d1"] + 1) % nranks
        parent_child_type1 = tree1["parent_child_type"]
    else:
        # Mirror logic
        mirrorrank = nranks - 1 - rank
        tree1 = get_nccl_btree(nranks, mirrorrank)
        s1 = -1 if tree1["u"] == -1 else nranks - 1 - tree1["u"]
        d1_0 = -1 if tree1["d0"] == -1 else nranks - 1 - tree1["d0"]
        d1_1 = -1 if tree1["d1"] == -1 else nranks - 1 - tree1["d1"]
        parent_child_type1 = tree1["parent_child_type"]

    return {
        "s0": s0,
        "d0_0": d0_0,
        "d0_1": d0_1,
        "parent_child_type0": parent_child_type0,
        "s1": s1,
        "d1_0": d1_0,
        "d1_1": d1_1,
        "parent_child_type1": parent_child_type1
    }

def apply_user_config(yaml_file, events_parallel_group, comm_init_events, comm_Info):
    num_gpus_traced = 0
    gpu_events_parallel_group = {}
    gpu_comm_init_events = {}

    for goal_rank, goal_events in comm_init_events.items():
        num_gpus_traced += len(goal_events)
        for gpuId, gpu_events in goal_events.items():
            gpu_comm_init_events[gpuId] = gpu_events

    for goal_rank, goal_events in events_parallel_group.items():
        for gpuId, gpu_events in goal_events.items():
            gpu_events_parallel_group[gpuId] = gpu_events

    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        if config is not None:
            if config.get('num_of_nodes') is not None and config.get('num_gpus_per_node') is not None:
                num_nodes = config['num_of_nodes']
                num_gpus_per_node = config['num_gpus_per_node']
                num_gpus = num_nodes * num_gpus_per_node

                assert num_gpus == num_gpus_traced, 'The number of gpus you configure is not equal to the number of gpus used in tracing'

                gpuId = 0
                gpuId_to_goal = {}
                for goal_rank in range(0, num_nodes):
                    for local_gpu_id in range(0, num_gpus_per_node):
                        gpuId_to_goal[gpuId] = goal_rank
                        gpuId += 1

            elif config.get('num_gpus_list') is not None:
                num_gpus_list = config['num_gpus_list']
                num_nodes = len(num_gpus_list)
                num_gpus = 0
                for num_gpus_node in num_gpus_list:
                    num_gpus += num_gpus_node

                assert num_gpus == num_gpus_traced, 'The number of gpus you configure is not equal to the number of gpus used in tracing'

                gpuId = 0
                gpuId_to_goal = {}
                for goal_rank in range(0, num_nodes):
                    for local_gpu_id in range(0, num_gpus_list[goal_rank]):
                        gpuId_to_goal[gpuId] = goal_rank
                        gpuId += 1

            events_parallel_group = {}
            comm_init_events = {}
            for gpuId in range(0, num_gpus):
                goal_rank = gpuId_to_goal[gpuId]
                if goal_rank not in events_parallel_group:
                    events_parallel_group[goal_rank] = {}
                if goal_rank not in comm_init_events:
                    comm_init_events[goal_rank] = {}
                events_parallel_group[goal_rank][gpuId] = gpu_events_parallel_group[gpuId]
                comm_init_events[goal_rank][gpuId] = gpu_comm_init_events[gpuId]

            for commId in comm_Info.keys():
                for rank in comm_Info[commId]["rank_To_rankInfo"].keys():
                    comm_Info[commId]["rank_To_rankInfo"][rank]["goal_rank"] = gpuId_to_goal[comm_Info[commId]["rank_To_rankInfo"][rank]["gpuId"]]
                    comm_Info[commId]["rank_To_rankInfo"][rank].pop("host_name", None)

            ## Restructure the Ring/Tree topology
            for commId in comm_Info.keys():
                comm_config = {}
                for rank in comm_Info[commId]["rank_To_rankInfo"].keys():
                    gpuId = comm_Info[commId]["rank_To_rankInfo"][rank]["gpuId"]
                    goal_rank = comm_Info[commId]["rank_To_rankInfo"][rank]["goal_rank"]

                    if goal_rank not in comm_config:
                        comm_config[goal_rank] = []

                    comm_config[goal_rank].append(gpuId)

                sorted_goal_rank = sorted(comm_config.keys())  ## Build chain within the node from lower gpuId to higher gpuId
                sorted_comm_config = {goal_rank: sorted(comm_config[goal_rank]) for goal_rank in sorted_goal_rank}
                comm_config = sorted_comm_config
                # print(comm_config)

                ## Build Ring/Tree between nodes
                topo_info = {}
                for goal_rank in comm_config.keys():
                    topo_info[goal_rank] = {}
                    topo_info[goal_rank]['gpu_list'] = comm_config[goal_rank]
                    topo_info[goal_rank]['Ring'] = {}
                    topo_info[goal_rank]['Tree_0'] = {}
                    topo_info[goal_rank]['Tree_1'] = {}

                goal_list = list(comm_config.keys())
                num_goal_rank = len(goal_list)
                for i in range(num_goal_rank):
                    goal_rank = goal_list[i]
                    topo_info[goal_rank]['Ring']['next_goal_rank'] = goal_list[(i + 1) % num_goal_rank]
                    topo_info[goal_rank]['Ring']['last_goal_rank'] = goal_list[(i - 1) % num_goal_rank]

                    DTree = get_nccl_dtree(num_goal_rank, i)

                    topo_info[goal_rank]['Tree_0']['parent_goal_rank'] = goal_list[DTree['s0']] if DTree['s0'] != -1 else -1
                    topo_info[goal_rank]['Tree_0']['child_1_goal_rank'] = goal_list[DTree['d0_0']] if DTree['d0_0'] != -1 else -1
                    topo_info[goal_rank]['Tree_0']['child_2_goal_rank'] = goal_list[DTree['d0_1']] if DTree['d0_1'] != -1 else -1

                    topo_info[goal_rank]['Tree_1']['parent_goal_rank'] = goal_list[DTree['s1']] if DTree['s1'] != -1 else -1
                    topo_info[goal_rank]['Tree_1']['child_1_goal_rank'] = goal_list[DTree['d1_0']] if DTree['d1_0'] != -1 else -1
                    topo_info[goal_rank]['Tree_1']['child_2_goal_rank'] = goal_list[DTree['d1_1']] if DTree['d1_1'] != -1 else -1

                # print(topo_info)

                gpuId_To_rank = comm_Info[commId]["gpuId_To_rank"]
                nranks = comm_Info[commId]["nranks"]
                # for rank in comm_Info[commId]["rank_To_rankInfo"].keys():
                #     gpuId = comm_Info[commId]["rank_To_rankInfo"][rank]["gpuId"]
                #     goal_rank = comm_Info[commId]["rank_To_rankInfo"][rank]["goal_rank"]

                for goal_rank in topo_info.keys():
                    gpuId_list = topo_info[goal_rank]['gpu_list']
                    for i in range(len(gpuId_list)):
                        gpuId = gpuId_list[i]
                        rank = gpuId_To_rank[gpuId]

                        for ring_topo in comm_Info[commId]["rank_To_rankInfo"][rank]['channel_info']["Ring"]:  ## Restructure Ring Topology
                            last_ring_goal_rank = topo_info[goal_rank]['Ring']['last_goal_rank']
                            next_ring_goal_rank = topo_info[goal_rank]['Ring']['next_goal_rank']
                            ring_topo["previous_rank"] = gpuId_list[i - 1] if i - 1 >= 0 else topo_info[last_ring_goal_rank]['gpu_list'][-1]
                            ring_topo["previous_rank"] = gpuId_To_rank[ring_topo["previous_rank"]]
                            ring_topo["next_rank"] = gpuId_list[i + 1] if i + 1 <= len(gpuId_list) -1 else topo_info[next_ring_goal_rank]['gpu_list'][0]
                            ring_topo["next_rank"] = gpuId_To_rank[ring_topo["next_rank"]]

                        for tree_id, tree_topo in enumerate(comm_Info[commId]["rank_To_rankInfo"][rank]['channel_info']["Tree"]):
                            tree_topo['parent_rank'] = "-1"
                            tree_topo['child_1_rank'] = "-1"
                            tree_topo['child_2_rank'] = "-1"
                            tree_topo['child_3_rank'] = "-1"

                            if tree_id % 2 == 0:  ## use Tree_0
                                parent_tree_goal_rank = topo_info[goal_rank]['Tree_0']['parent_goal_rank']
                                child_1_tree_goal_rank = topo_info[goal_rank]['Tree_0']['child_1_goal_rank']
                                child_2_tree_goal_rank = topo_info[goal_rank]['Tree_0']['child_2_goal_rank']

                            elif tree_id % 2 == 1:  ## use Tree_1
                                parent_tree_goal_rank = topo_info[goal_rank]['Tree_1']['parent_goal_rank']
                                child_1_tree_goal_rank = topo_info[goal_rank]['Tree_1']['child_1_goal_rank']
                                child_2_tree_goal_rank = topo_info[goal_rank]['Tree_1']['child_2_goal_rank']

                            child_rank_list = []

                            if (i - 1) >= 0:
                                child_gpuId = gpuId_list[i - 1]
                                child_rank = gpuId_To_rank[child_gpuId]
                                child_rank_list.append(child_rank)

                            else:
                                if child_1_tree_goal_rank != -1:
                                    child_gpuId = topo_info[child_1_tree_goal_rank]['gpu_list'][-1]
                                    child_rank = gpuId_To_rank[child_gpuId]
                                    child_rank_list.append(child_rank)

                                if child_2_tree_goal_rank != -1:
                                    child_gpuId = topo_info[child_2_tree_goal_rank]['gpu_list'][-1]
                                    child_rank = gpuId_To_rank[child_gpuId]
                                    child_rank_list.append(child_rank)

                            for child_id in range(len(child_rank_list)):
                                if child_id == 0: 
                                    tree_topo['child_1_rank'] = child_rank_list[child_id]

                                elif child_id == 1: 
                                    tree_topo['child_2_rank'] = child_rank_list[child_id]

                                elif child_id == 2: 
                                    tree_topo['child_3_rank'] = child_rank_list[child_id]

                            if (i + 1) <= len(gpuId_list) -1:
                                parent_gpuId = gpuId_list[i + 1]
                                parent_rank = gpuId_To_rank[parent_gpuId]
                                tree_topo['parent_rank'] = parent_rank

                            else:
                                if parent_tree_goal_rank != -1:
                                    parent_gpuId = topo_info[parent_tree_goal_rank]['gpu_list'][0]
                                    parent_rank = gpuId_To_rank[parent_gpuId]
                                    tree_topo['parent_rank'] = parent_rank


        else:
            print('no configuration in config file')

    return events_parallel_group, comm_init_events, comm_Info
