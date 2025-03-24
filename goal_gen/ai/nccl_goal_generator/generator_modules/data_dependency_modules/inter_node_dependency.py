from .utils import modRanks, div_up, get_event_type
from .intra_node_gpu_transfer_time import get_intra_node_gpu_transfer_time
from .reduction_copy_time import get_reduction_time, get_copy_time

def get_inter_node_microevents_dependency(nccl_group_events, comm_init_events, comm_info,
                                          SendRecvEvents_To_TaskCounter,
                                          goal_file_name, profile_interval={},
                                          zero_red_copy=False, unique_nic=False):
    num_ranks = len(nccl_group_events)
    if zero_red_copy:
        print("[INFO] Zero reduction copy and data transfer time is enabled")
    
    if unique_nic:
        print("[INFO] Assigning unique NIC for communication for each GPU")

    max_comp_time = 0
    # task_counter = 0
    with open(goal_file_name, 'w') as file:
        file.write(f"num_ranks {num_ranks}\n")

        for goal_rank in range(num_ranks):
            task_counter = 0

            file.write(f"\nrank {goal_rank}")
            file.write(" {\n")

            cpu_counter_start = 0
            cpu_counter_end = 0
            cpu_counter = 0
            gpuId_commId_cpu_counter = {}

            goal_events = nccl_group_events[goal_rank]
            task_counter += 1
            file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n") ## Start point of the node
            node_start_calc_id = task_counter
            
            task_counter += 1
            file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n") ## End point of the node
            node_end_calc_id = task_counter

            for gpu_idx, (gpuId, gpu_events) in enumerate(goal_events.items()):
                if unique_nic:
                    # Each GPU has a unique NIC id
                    nicId = gpu_idx
                else:
                    # All GPUs share the same NIC id
                    nicId = 0

                gpuId_commId_cpu_counter[gpuId] = {}
                
                gpu_all_stream_start_time = None
                if gpuId in profile_interval:
                    gpu_all_stream_start_time = profile_interval[gpuId]["start"]
                    gpu_all_stream_end_time = profile_interval[gpuId]["end"]
                    print(f"[DEBUG] Profiling interval for GPU {gpuId}: [{gpu_all_stream_start_time},{gpu_all_stream_end_time}] ({(gpu_all_stream_end_time - gpu_all_stream_start_time) / 1e9:.3f} s)")
                else:
                    for streamId, stream_events in gpu_events.items():
                        if gpu_all_stream_start_time is None:
                            gpu_all_stream_start_time = stream_events[0]['ts_group_gpu_start']
                        else:
                            gpu_all_stream_start_time = min(gpu_all_stream_start_time, stream_events[0]['ts_group_gpu_start'])
                    gpu_all_stream_end_time = float('inf')
                profiling_interval = gpu_all_stream_end_time - gpu_all_stream_start_time
                exposed_comm_time = 0
                # calc_time = 0
                for streamId, stream_events in gpu_events.items():
                    cpu_counter_start = cpu_counter_end + 1
                    cpu_counter = cpu_counter_start
                    cpu_counter_end = cpu_counter

                    last_group_event_end_time =  gpu_all_stream_start_time
                    last_group_event_end_id = node_start_calc_id
                    for group_event_index, group_event in enumerate(stream_events):
                        if group_event['ts_group_gpu_start'] < gpu_all_stream_start_time:
                            continue

                        if group_event["ts_group_gpu_start"] >= gpu_all_stream_end_time:

                            file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")
                            break
                        
                        exposed_comm_time += group_event['ts_group_gpu_end'] - group_event['ts_group_gpu_start']
                        launched = 0
                        cpu_counter = cpu_counter_start

                        task_counter += 1
                        file.write(f"l{task_counter}: calc {group_event['ts_group_gpu_start'] - last_group_event_end_time} cpu {cpu_counter}\n")  ## Former calc between first group host event start and last group gpu event end
                        # calc_time += group_event['ts_group_gpu_start'] - last_group_event_end_time
                        file.write(f"l{task_counter} requires l{last_group_event_end_id}\n")
                        group_event_start_calc_id = task_counter

                        task_counter += 1
                        file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n")  ## End calc of the parallel group of events
                        group_event_end_calc_id = task_counter
                        last_group_event_end_time = group_event['ts_group_gpu_end']
                        last_group_event_end_id = task_counter

                        for event_index, event in enumerate(group_event['events']):
                            # file.write(f" START {event['event_type']}\n")
                            if event['event_type'] == 'Send' or event['event_type'] == 'Recv':
                                commId = event['commId']
                                p2p_event_type = event['event_type']
                                p2p_peer_Ix = event['peer_rank']
                                gpuId_peer = comm_info[commId]['rank_To_rankInfo'][p2p_peer_Ix]['gpuId']
                                goal_rank_peer = comm_info[commId]['rank_To_rankInfo'][p2p_peer_Ix]['goal_rank']
                                p2p_seq = event['seq']

                                if commId not in gpuId_commId_cpu_counter[gpuId]:
                                    gpuId_commId_cpu_counter[gpuId][commId] = []

                                if event_index >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                    cpu_counter_end += 1
                                    cpu_counter = cpu_counter_end
                                    gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                else:
                                    cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][event_index]

                                if launched == 0:
                                    task_counter += 1
                                    file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n")  ## Former calc between nccl kernel launch end and host event start
                                    file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                    p2p_group_start_calc_id = task_counter

                                    task_counter += 1
                                    file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n")
                                    file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")
                                    p2p_group_end_calc_id = task_counter

                                    launched = 1

                                p2p_index = {} 
                                p2p_index[p2p_peer_Ix] = 0 
                                channel_id = 0

                                proto = event['protocol']
                                chunkSize = event['chunkSize']
                                count = event['count']

                                # if proto == '0': ## LL
                                #     chunkSize //= 2
                                #     for elemOffset in range(0, count, chunkSize):
                                #         nelem = int(min(chunkSize, count - elemOffset))
                                #         nelem = 0 if nelem < 0 else nelem

                                #         task_counter += 1
                                #         tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                #         if p2p_event['event_type'] == 'Send':
                                #             file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {p2p_event['peer_rank']}\n")
                                #         elif p2p_event['event_type'] == 'Recv':
                                #             file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {p2p_event['peer_rank']}\n")
                                #         file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                                #         file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")

                                if proto == '2': ## Simple
                                    for elemOffset in range(0, count, chunkSize):
                                        nelem = int(min(chunkSize, count - elemOffset))
                                        nelem = 0 if nelem < 0 else nelem

                                        task_counter += 1
                                        if p2p_event_type == 'Send':
                                            if goal_rank_peer != goal_rank:
                                                tag = str(p2p_index[p2p_peer_Ix]) + str(channel_id).zfill(2) + str(p2p_seq).zfill(4) + str(get_event_type(p2p_event_type)).zfill(1) + str(event['comm_index']).zfill(2)
                                                file.write(f"l{task_counter}: send {max(1, nelem)}b to {goal_rank_peer} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                p2p_index[p2p_peer_Ix] += 1
                                            else:
                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(nelem, 'Send', zero_red_copy)} cpu {cpu_counter}\n")

                                        elif p2p_event_type == 'Recv':
                                            if goal_rank_peer != goal_rank:
                                                tag = str(p2p_index[p2p_peer_Ix]) + str(channel_id).zfill(2) + str(p2p_seq).zfill(4) + str(get_event_type(p2p_event_type)).zfill(1) + str(event['comm_index']).zfill(2)
                                                file.write(f"l{task_counter}: recv {max(1, nelem)}b from {goal_rank_peer} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                p2p_index[p2p_peer_Ix] += 1
                                            else:
                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(nelem, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")

                                        file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                                        file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")
                            
                            else:
                                commId = event['commId']
                                nranks = comm_info[commId]['nranks']

                                if commId not in gpuId_commId_cpu_counter[gpuId]:
                                    gpuId_commId_cpu_counter[gpuId][commId] = []

                                if 0 >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                    cpu_counter_end += 1
                                    cpu_counter = cpu_counter_end
                                    gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                else:
                                    cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][0] 
                                
                                if launched == 0:
                                    task_counter += 1
                                    file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n")  ## Former calc between nccl kernel launch end and host event start
                                    file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                    gpu_event_start_calc_id = task_counter

                                    task_counter += 1
                                    file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n")  ## end calc of a gpu event
                                    file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")          
                                    gpu_event_end_calc_id = task_counter     

                                    launched = 1

                                if event['event_type'] == 'AllReduce':
                                    algo = event['algorithm']  ## NCCL_ALGO_TREE: 0, NCCL_ALGO_RING: 1
                                    proto = event['protocol']  ## NCCL_PROTO_LL: 0, NCCL_PROTO_LL128: 1, NCCL_PROTO_SIMPLE: 2
                                    type_size = event['type_size']
                                    chunkSteps = event['chunkSteps']
                                    sliceSteps = event['sliceSteps']
                                    stepSize = event['stepSize']

                                    if algo == '1': ## Ring
                                        ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                        channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                        elems = event['elems']
                                        for channel_id, elem in enumerate(elems):
                                            send_index = {}
                                            recv_index = {}

                                            nranks = comm_info[event['commId']]['nranks']  ## 2
                                            prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                            recv_index[prevIx] = 0
                                            gpuId_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['gpuId']
                                            goal_rank_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['goal_rank']
                                            nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                            send_index[nextIx] = 0
                                            gpuId_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['gpuId']
                                            goal_rank_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['goal_rank']
                                            
                                            chunkCount = elem['chunkCount']
                                            gridOffset = elem['workOffset']
                                            channelCount = elem['workCount']
                                            lastChunkCount = elem['lastChunkCount']
                                            loopCount = nranks * chunkCount

                                            if commId not in gpuId_commId_cpu_counter[gpuId]:
                                                gpuId_commId_cpu_counter[gpuId][commId] = []

                                            if channel_id >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                                cpu_counter_end += 1
                                                cpu_counter = cpu_counter_end
                                                gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                            else:
                                                cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][channel_id] 

                                            for elemOffset in range(0, channelCount, loopCount):
                                                remCount = channelCount - elemOffset
                                                if (remCount < loopCount):
                                                    chunkCount = lastChunkCount
                                                
                                                ## step 0: Send
                                                chunk = modRanks(int(ringIx) + int(nranks) - 1, int(nranks))
                                                chunkOffset = chunk * chunkCount
                                                # offset = gridOffset + elemOffset + chunkOffset
                                                nelem = int(min(chunkCount, remCount - chunkOffset))
                                                nelem = 0 if nelem < 0 else nelem
                                                # prims.send(offset, nelem)
                                                if proto == '0':
                                                    # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break
                                                        
                                                ## Step 1 to step (k - 2): RecvReduceSend
                                                for j in range(2, nranks):
                                                    chunk = modRanks(int(ringIx) + int(nranks) - j, int(nranks))
                                                    chunkOffset = chunk * chunkCount
                                                    # offset = gridOffset + elemOffset + chunkOffset
                                                    nelem = int(min(chunkCount, remCount - chunkOffset))
                                                    nelem = 0 if nelem < 0 else nelem
                                                    # prims.recvReduceSend(offset, nelem)

                                                    if proto == '0':
                                                        task_counter += 1
                                                        if goal_rank_prev != goal_rank:
                                                            tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[prevIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                    elif proto == '2':
                                                        sliceSize = stepSize * sliceSteps
                                                        SlicePerChunk = chunkSteps // sliceSteps
                                                        sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                        slice = 0
                                                        offset = 0

                                                        if offset < nelem:
                                                            while True:
                                                                sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                                task_counter += 1
                                                                if goal_rank_prev != goal_rank:
                                                                    tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    recv_index[prevIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                
                                                                task_counter += 1
                                                                if goal_rank_next != goal_rank:
                                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                                slice += 1
                                                                offset += sliceSize

                                                                if not (slice < SlicePerChunk and offset < nelem):
                                                                    break

                                                ## Step (k - 1): RecvReduceCopySend
                                                chunk = int(ringIx) + 0  # 0
                                                chunkOffset = chunk * chunkCount  ## 0
                                                # offset = gridOffset + elemOffset + chunkOffset  ## 0
                                                nelem = int(min(chunkCount, remCount - chunkOffset))  ## min(524288， 1024 - 524288)
                                                nelem = 0 if nelem < 0 else nelem
                                                # prims.directRecvReduceCopySend(offset, offset, nelem, /*postOp=*/true)

                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1
                                                    
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy) + get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy) + get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break
                                                            
                                                ## Step k to step (2k - 3): RecvCopySend
                                                for j in range(1, nranks - 1):
                                                    chunk = modRanks(int(ringIx) + int(nranks) - j, int(nranks))
                                                    chunkOffset = chunk * chunkCount
                                                    # offset = gridOffset + elemOffset + chunkOffset
                                                    nelem = int(min(chunkCount, remCount - chunkOffset))
                                                    nelem = 0 if nelem < 0 else nelem
                                                    # prims.directRecvCopySend(offset, nelem)

                                                    if proto == '0':
                                                        task_counter += 1
                                                        if goal_rank_prev != goal_rank:
                                                            tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[prevIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                    elif proto == '2':
                                                        sliceSize = stepSize * sliceSteps
                                                        SlicePerChunk = chunkSteps // sliceSteps
                                                        sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                        slice = 0
                                                        offset = 0

                                                        if offset < nelem:
                                                            while True:
                                                                sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                                task_counter += 1
                                                                if goal_rank_prev != goal_rank:
                                                                    tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    recv_index[prevIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                
                                                                task_counter += 1
                                                                if goal_rank_next != goal_rank:
                                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                
                                                                slice += 1
                                                                offset += sliceSize

                                                                if not (slice < SlicePerChunk and offset < nelem):
                                                                    break

                                                ## Step (2k - 2): Recv
                                                chunk = modRanks(int(ringIx) + 1, int(nranks))
                                                chunkOffset = chunk * chunkCount
                                                # offset = gridOffset + elemOffset + chunkOffset
                                                nelem = int(min(chunkCount, remCount - chunkOffset))
                                                nelem = 0 if nelem < 0 else nelem
                                                # prims.directRecv(offset, nelem)

                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                recv_index[prevIx] += 1
                                                            
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break
                                    
                                    elif algo == '0': ## Tree AllReduce
                                        myIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                        channel_info = comm_info[commId]['rank_To_rankInfo'][myIx]['channel_info']['Tree']

                                        elems = event['elems']
                                        for channel_id, elem in enumerate(elems):
                                            send_index = {}
                                            recv_index = {}

                                            nranks = comm_info[event['commId']]['nranks']  ## 2
                                            child_1_Ix = channel_info[channel_id]['child_1_rank']  ## local rank index in the communicator
                                            if child_1_Ix != '-1':
                                                send_index[child_1_Ix] = 0
                                                recv_index[child_1_Ix] = 0
                                                gpuId_child_1 = comm_info[commId]['rank_To_rankInfo'][child_1_Ix]['gpuId']
                                                goal_rank_child_1 = comm_info[commId]['rank_To_rankInfo'][child_1_Ix]['goal_rank']

                                            child_2_Ix = channel_info[channel_id]['child_2_rank']  ## local rank index in the communicator
                                            if child_2_Ix != '-1':
                                                send_index[child_2_Ix] = 0
                                                recv_index[child_2_Ix] = 0
                                                gpuId_child_2 = comm_info[commId]['rank_To_rankInfo'][child_2_Ix]['gpuId']
                                                goal_rank_child_2 = comm_info[commId]['rank_To_rankInfo'][child_2_Ix]['goal_rank']

                                            child_3_Ix = channel_info[channel_id]['child_3_rank']  ## local rank index in the communicator
                                            if child_3_Ix != '-1':
                                                send_index[child_3_Ix] = 0
                                                recv_index[child_3_Ix] = 0
                                                gpuId_child_3 = comm_info[commId]['rank_To_rankInfo'][child_3_Ix]['gpuId']
                                                goal_rank_child_3 = comm_info[commId]['rank_To_rankInfo'][child_3_Ix]['goal_rank']
                                                
                                            parent_Ix = channel_info[channel_id]['parent_rank']  ## local rank index in the communicator
                                            if parent_Ix != '-1':
                                                send_index[parent_Ix] = 0
                                                recv_index[parent_Ix] = 0
                                                gpuId_parent = comm_info[commId]['rank_To_rankInfo'][parent_Ix]['gpuId']
                                                goal_rank_parent = comm_info[commId]['rank_To_rankInfo'][parent_Ix]['goal_rank']

                                            if commId not in gpuId_commId_cpu_counter[gpuId]:
                                                gpuId_commId_cpu_counter[gpuId][commId] = []

                                            if channel_id >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                                cpu_counter_end += 1
                                                cpu_counter = cpu_counter_end
                                                gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                            else:
                                                cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][channel_id] 
                                            
                                            chunkCount = elem['chunkCount']
                                            gridOffset = elem['workOffset']
                                            channelCount = elem['workCount']
                                            lastChunkCount = elem['lastChunkCount']

                                            if parent_Ix == '-1':  #  Top-most rank: RecvReduceCopySend from child to child
                                                for elemOffset in range(0, channelCount, chunkCount):
                                                    nelem = int(min(chunkCount, channelCount - elemOffset))
                                                    nelem = 0 if nelem < 0 else nelem
                                                    if proto == '0':
                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy) + get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                        calc_task_id = task_counter

                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']
                                                                
                                                                task_counter += 1
                                                                if goal_rank != goal_rank_child:
                                                                    tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    recv_index[child_Ix] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")

                                                                task_counter += 1
                                                                if goal_rank != goal_rank_child:
                                                                    tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[child_Ix] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                    elif proto == '2':
                                                        sliceSize = stepSize * sliceSteps
                                                        SlicePerChunk = chunkSteps // sliceSteps
                                                        sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                        slice = 0
                                                        offset = 0

                                                        if offset < nelem:
                                                            while True:
                                                                sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy) + get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                                calc_task_id = task_counter

                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                        goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']
                                                                        
                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                            recv_index[child_Ix] += 1

                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")

                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                            send_index[child_Ix] += 1

                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                
                                                                slice += 1
                                                                offset += sliceSize

                                                                if not (slice < SlicePerChunk and offset < nelem):
                                                                    break

                                            elif child_1_Ix == '-1': ## Bottom-most rank: Send to parent && Recv from parent
                                                for elemOffset in range(0, channelCount, chunkCount):
                                                    nelem = int(min(chunkCount, channelCount - elemOffset))
                                                    nelem = 0 if nelem < 0 else nelem
                                                    if proto == '0':
                                                        task_counter += 1  ## Send
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(send_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[parent_Ix] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        task_counter += 1  ## Recv
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            recv_index[parent_Ix] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                    elif proto == '2':
                                                        sliceSize = stepSize * sliceSteps
                                                        SlicePerChunk = chunkSteps // sliceSteps
                                                        sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                        slice = 0
                                                        offset = 0

                                                        if offset < nelem:
                                                            while True:
                                                                sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                                task_counter += 1  ## Send
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(send_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                                task_counter += 1  ## Recv
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    recv_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                                slice += 1
                                                                offset += sliceSize

                                                                if not (slice < SlicePerChunk and offset < nelem):
                                                                    break

                                            else: ## Middle rank: RecvReduceSend from child to parent && RecvCopySend from parent to child
                                                for elemOffset in range(0, channelCount, chunkCount):
                                                    nelem = int(min(chunkCount, channelCount - elemOffset))
                                                    nelem = 0 if nelem < 0 else nelem
                                                    if proto == '0':
                                                        ## RecvReduceSend
                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                        calc_task_id = task_counter

                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                task_counter += 1
                                                                if goal_rank_child != goal_rank:
                                                                    tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    recv_index[child_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                    
                                                        task_counter += 1
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(send_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[parent_Ix] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        ## RecvCopySend
                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                        calc_task_id = task_counter

                                                        task_counter += 1
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[parent_Ix] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        
                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                task_counter += 1
                                                                if goal_rank_child != goal_rank:
                                                                    tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[child_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                    elif proto == '2':
                                                        sliceSize = stepSize * sliceSteps
                                                        SlicePerChunk = chunkSteps // sliceSteps
                                                        sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                        slice = 0
                                                        offset = 0

                                                        if offset < nelem:
                                                            while True:
                                                                sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                                ## RecvReduceSend
                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                                calc_task_id = task_counter

                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                        goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                            recv_index[child_Ix] += 1
                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                            
                                                                task_counter += 1
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(send_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                                ## RecvCopySend
                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                                calc_task_id = task_counter

                                                                task_counter += 1
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_parent} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    recv_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                
                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                        goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_child} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                            send_index[child_Ix] += 1
                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                
                                                                slice += 1
                                                                offset += sliceSize

                                                                if not (slice < SlicePerChunk and offset < nelem):
                                                                    break

                                elif event['event_type'] == 'Broadcast':
                                    algo = event['algorithm']  ## NCCL_ALGO_TREE: 0, NCCL_ALGO_RING: 1, broadcast only has Ring
                                    proto = event['protocol']  ## NCCL_PROTO_LL: 0, NCCL_PROTO_LL128: 1, NCCL_PROTO_SIMPLE: 2
                                    
                                    root_rank = event['root_rank']

                                    type_size = event['type_size']
                                    chunkSteps = event['chunkSteps']
                                    sliceSteps = event['sliceSteps']
                                    stepSize = event['stepSize']

                                    ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        send_index = {}
                                        recv_index = {}

                                        nranks = comm_info[event['commId']]['nranks']  ## 2
                                        prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                        recv_index[prevIx] = 0
                                        gpuId_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['gpuId']
                                        goal_rank_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['goal_rank']
                                        nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                        send_index[nextIx] = 0
                                        gpuId_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['gpuId']
                                        goal_rank_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['goal_rank']

                                        if commId not in gpuId_commId_cpu_counter[gpuId]:
                                            gpuId_commId_cpu_counter[gpuId][commId] = []

                                        if channel_id >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                            cpu_counter_end += 1
                                            cpu_counter = cpu_counter_end
                                            gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                        else:
                                            cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][channel_id]
                                        
                                        chunkCount = elem['chunkCount']
                                        gridOffset = elem['workOffset']
                                        channelCount = elem['workCount']
                                        lastChunkCount = elem['lastChunkCount']
                                        count = elem['count']
                                        sendbuff = elem['sendbuff']
                                        recvbuff = elem['recvbuff']
                                        loopCount = nranks * chunkCount

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            # offset = gridOffset + elemOffset
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            if (ringIx == root_rank):  ## Send
                                                if proto == '0':
                                                    # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                    if sendbuff == recvbuff:  ## In-Place: Send
                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, div_up(nelem, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter}nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                    else:  ## CopySend
                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, div_up(nelem, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter}nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset
                                                            
                                                            if sendbuff == recvbuff:  ## In-Place: Send
                                                                task_counter += 1
                                                                if goal_rank_next != goal_rank:
                                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, sliceSize)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                            else:  ## CopySend
                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto, zero_red_copy)}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                                task_counter += 1
                                                                if goal_rank_next != goal_rank:
                                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {max(1, sliceSize)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter -1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            elif nextIx == root_rank: ## Recv
                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                recv_index[prevIx] += 1
                                                            
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break
                                                
                                            else:  ## RecvCopySend
                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                elif event['event_type'] == 'AllGather':
                                    algo = event['algorithm']  ## NCCL_ALGO_TREE: 0, NCCL_ALGO_RING: 1
                                    proto = event['protocol']  ## NCCL_PROTO_LL: 0, NCCL_PROTO_LL128: 1, NCCL_PROTO_SIMPLE: 2
                                    type_size = event['type_size']
                                    chunkSteps = event['chunkSteps']
                                    sliceSteps = event['sliceSteps']
                                    stepSize = event['stepSize']

                                    # if algo == '1': ## Ring AllGather
                                    ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        send_index = {}
                                        recv_index = {}

                                        nranks = comm_info[event['commId']]['nranks']
                                        prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                        recv_index[prevIx] = 0
                                        gpuId_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['gpuId']
                                        goal_rank_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['goal_rank']
                                        nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                        send_index[nextIx] = 0
                                        gpuId_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['gpuId']
                                        goal_rank_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['goal_rank']

                                        if commId not in gpuId_commId_cpu_counter[gpuId]:
                                            gpuId_commId_cpu_counter[gpuId][commId] = []

                                        if channel_id >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                            cpu_counter_end += 1
                                            cpu_counter = cpu_counter_end
                                            gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                        else:
                                            cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][channel_id] 
                                        
                                        chunkCount = elem['chunkCount']
                                        gridOffset = elem['workOffset']
                                        channelCount = elem['workCount']
                                        lastChunkCount = elem['lastChunkCount']
                                        count = elem['count']
                                        sendbuff = elem['sendbuff']
                                        recvbuff = elem['recvbuff']

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            ## step 0: Send
                                            if proto == '0':
                                                # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                if sendbuff == recvbuff + int(ringIx) * count: ## In-Place: Send
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                else:  ## CopySend
                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                            elif proto == '2':
                                                sliceSize = stepSize * sliceSteps
                                                SlicePerChunk = chunkSteps // sliceSteps
                                                sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                slice = 0
                                                offset = 0

                                                if offset < nelem:
                                                    while True:
                                                        sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                        if sendbuff == recvbuff + int(ringIx) * count: ## In-Place: Send
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        else:  ## CopySend
                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break
                                                           
                                            ## Step 1 to step (k - 2): RecvCopySend
                                            for j in range(1, nranks - 1):
                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            ## Step (k - 1): Recv
                                            if proto == '0':
                                                task_counter += 1
                                                if goal_rank_prev != goal_rank:
                                                    tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: recv {max(1, div_up(nelem, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    recv_index[prevIx] += 1
                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                            elif proto == '2':
                                                sliceSize = stepSize * sliceSteps
                                                SlicePerChunk = chunkSteps // sliceSteps
                                                sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                slice = 0
                                                offset = 0

                                                if offset < nelem:
                                                    while True:
                                                        sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                        task_counter += 1
                                                        if goal_rank_prev != goal_rank:
                                                            tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {max(1, sliceSize)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            recv_index[prevIx] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break 

                                elif event['event_type'] == 'ReduceScatter':
                                    algo = event['algorithm']  ## NCCL_ALGO_TREE: 0, NCCL_ALGO_RING: 1
                                    proto = event['protocol']  ## NCCL_PROTO_LL: 0, NCCL_PROTO_LL128: 1, NCCL_PROTO_SIMPLE: 2
                                    type_size = event['type_size']
                                    chunkSteps = event['chunkSteps']
                                    sliceSteps = event['sliceSteps']
                                    stepSize = event['stepSize']

                                    # if algo == '1': ## Ring ReduceScatter
                                    ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        send_index = {}
                                        recv_index = {}

                                        nranks = comm_info[event['commId']]['nranks']
                                        prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                        recv_index[prevIx] = 0
                                        gpuId_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['gpuId']
                                        goal_rank_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['goal_rank']
                                        nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                        send_index[nextIx] = 0
                                        gpuId_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['gpuId']
                                        goal_rank_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['goal_rank']

                                        if commId not in gpuId_commId_cpu_counter[gpuId]:
                                            gpuId_commId_cpu_counter[gpuId][commId] = []

                                        if channel_id >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                            cpu_counter_end += 1
                                            cpu_counter = cpu_counter_end
                                            gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                        else:
                                            cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][channel_id] 
                                        
                                        chunkCount = elem['chunkCount']
                                        gridOffset = elem['workOffset']
                                        channelCount = elem['workCount']
                                        lastChunkCount = elem['lastChunkCount']

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            ## step 0: Send
                                            if proto == '0':
                                                # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                task_counter += 1
                                                if goal_rank_next != goal_rank:
                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    send_index[nextIx] += 1
                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                            elif proto == '2':
                                                sliceSize = stepSize * sliceSteps
                                                SlicePerChunk = chunkSteps // sliceSteps
                                                sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                slice = 0
                                                offset = 0

                                                if offset < nelem:
                                                    while True:
                                                        sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter}nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break
                                                           
                                            ## Step 1 to step (k - 2): RecvReduceSend
                                            for j in range(1, nranks - 1):
                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            ## Step (k - 1): RecvReduceCopy
                                            if proto == '0':
                                                task_counter += 1
                                                if goal_rank_prev != goal_rank:
                                                    tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    recv_index[prevIx] += 1

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy) + get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy) + get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                            elif proto == '2':
                                                sliceSize = stepSize * sliceSteps
                                                SlicePerChunk = chunkSteps // sliceSteps
                                                sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                slice = 0
                                                offset = 0

                                                if offset < nelem:
                                                    while True:
                                                        sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                        task_counter += 1
                                                        if goal_rank_prev != goal_rank:
                                                            tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[prevIx] += 1

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy) + get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy) + get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break

                                elif event['event_type'] == 'Reduce':
                                    algo = event['algorithm']  ## NCCL_ALGO_TREE: 0, NCCL_ALGO_RING: 1, reduce only has Ring
                                    proto = event['protocol']  ## NCCL_PROTO_LL: 0, NCCL_PROTO_LL128: 1, NCCL_PROTO_SIMPLE: 2
                                    
                                    root_rank = event['root_rank']

                                    type_size = event['type_size']
                                    chunkSteps = event['chunkSteps']
                                    sliceSteps = event['sliceSteps']
                                    stepSize = event['stepSize']

                                    ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        send_index = {}
                                        recv_index = {}

                                        nranks = comm_info[event['commId']]['nranks']  ## 2
                                        prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                        recv_index[prevIx] = 0
                                        gpuId_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['gpuId']
                                        goal_rank_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['goal_rank']
                                        nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                        send_index[nextIx] = 0
                                        gpuId_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['gpuId']
                                        goal_rank_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['goal_rank']

                                        if commId not in gpuId_commId_cpu_counter[gpuId]:
                                            gpuId_commId_cpu_counter[gpuId][commId] = []

                                        if channel_id >= len(gpuId_commId_cpu_counter[gpuId][commId]):
                                            cpu_counter_end += 1
                                            cpu_counter = cpu_counter_end
                                            gpuId_commId_cpu_counter[gpuId][commId].append(cpu_counter)
                                        else:
                                            cpu_counter = gpuId_commId_cpu_counter[gpuId][commId][channel_id]
                                        
                                        chunkCount = elem['chunkCount']
                                        gridOffset = elem['workOffset']
                                        channelCount = elem['workCount']
                                        lastChunkCount = elem['lastChunkCount']
                                        count = elem['count']
                                        sendbuff = elem['sendbuff']
                                        recvbuff = elem['recvbuff']
                                        loopCount = nranks * chunkCount

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            # offset = gridOffset + elemOffset
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            if (prevIx == root_rank):  ## Send
                                                if proto == '0':
                                                    # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter}nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset
                                        
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            elif ringIx == root_rank: ## RecvReduceCopy
                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy) + get_copy_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy) + get_copy_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break
                                                
                                            else:  ## RecvReduceSend
                                                if proto == '0':
                                                    task_counter += 1
                                                    if goal_rank_prev != goal_rank:
                                                        tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {max(1, div_up(nelem * type_size, 8) * 16)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {max(1, div_up(nelem * type_size, 8) * 16)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                elif proto == '2':
                                                    sliceSize = stepSize * sliceSteps
                                                    SlicePerChunk = chunkSteps // sliceSteps
                                                    sliceSize = max(div_up(nelem, 16 * SlicePerChunk) * 16, sliceSize // 32)
                                                    slice = 0
                                                    offset = 0

                                                    if offset < nelem:
                                                        while True:
                                                            sliceSize = sliceSize if sliceSize < nelem-offset else nelem-offset

                                                            task_counter += 1
                                                            if goal_rank_prev != goal_rank:
                                                                tag = str(recv_index[prevIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {max(1, sliceSize * type_size)}b from {goal_rank_prev} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto, zero_red_copy)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {max(1, sliceSize * type_size)}b to {goal_rank_next} tag {tag} cpu {cpu_counter} nic {nicId}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send', zero_red_copy)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                else:
                                    task_counter += 1
                                    file.write(f"l{task_counter}: {event['event_type']} {event['data_size']} bytes comm {event['comm_index']} gpu {gpuId} stream {streamId}\n")  ## gpu event
                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                        
                            # file.write(f" END {event['event_type']}\n")
                        if group_event_index == len(stream_events) - 1:
                            file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")

                print(f"[DEBUG] Exposed comm time: {exposed_comm_time / 1e9:.5f} s")
                max_comp_time = max(max_comp_time, profiling_interval - exposed_comm_time)
                # print(f"[DEBUG] Calc time: {calc_time / 1e9:.5f} s")

            for gpuId, gpu_events in goal_events.items():
                if gpuId in profile_interval:
                    gpu_all_stream_start_time = profile_interval[gpuId]['start']
                    gpu_all_stream_end_time = profile_interval[gpuId]['end']
                else:
                    gpu_all_stream_start_time = 0
                    gpu_all_stream_end_time = float('inf')
                for streamId, stream_events in gpu_events.items():
                    for group_event_index, group_event in enumerate(stream_events):
                        if group_event["ts_group_gpu_start"] <= gpu_all_stream_start_time or \
                            group_event["ts_group_gpu_end"] >= gpu_all_stream_end_time:
                            continue

                        for event in group_event['events']:
                            if event['event_type'] == 'AllReduce' or event['event_type'] == 'Broadcast' or event['event_type'] == 'AllGather' or event['event_type'] == 'ReduceScatter' or event['event_type'] == 'Reduce':
                                algo = event['algorithm']
                                if algo == '1':  ## Ring
                                    commId = event['commId']
                                    ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                        gpuId_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['gpuId']
                                        goal_rank_prev = comm_info[commId]['rank_To_rankInfo'][prevIx]['goal_rank']
                                        nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                        gpuId_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['gpuId']
                                        goal_rank_next = comm_info[commId]['rank_To_rankInfo'][nextIx]['goal_rank']

                                        my_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx]
                                        prev_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank_prev][gpuId_prev][commId][event['event_type']][event['seq']][channel_id]['send'][ringIx]

                                        if goal_rank_prev == goal_rank:
                                            for i in range(len(my_event_task_counter)):
                                                my_receive_task_counter = my_event_task_counter[i]
                                                prev_send_task_counter = prev_event_task_counter[i]
                                                file.write(f"l{int(my_receive_task_counter)} requires l{int(prev_send_task_counter)}\n")
                                
                                elif algo == '0':  ## Tree
                                    commId = event['commId']
                                    myIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][myIx]['channel_info']['Tree']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        child_1_Ix = channel_info[channel_id]['child_1_rank']  ## local rank index in the communicator
                                        if child_1_Ix != '-1':
                                            gpuId_child_1 = comm_info[commId]['rank_To_rankInfo'][child_1_Ix]['gpuId']
                                            goal_rank_child_1 = comm_info[commId]['rank_To_rankInfo'][child_1_Ix]['goal_rank']
                                            
                                            my_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_1_Ix]
                                            child_1_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank_child_1][gpuId_child_1][commId][event['event_type']][event['seq']][channel_id]['send'][myIx]

                                            if goal_rank_child_1 == goal_rank:
                                                for i in range(len(my_event_task_counter)):
                                                    my_receive_task_counter = my_event_task_counter[i]
                                                    child_1_send_task_counter = child_1_event_task_counter[i]
                                                    file.write(f"l{int(my_receive_task_counter)} requires l{int(child_1_send_task_counter)}\n")

                                        child_2_Ix = channel_info[channel_id]['child_2_rank']  ## local rank index in the communicator
                                        if child_2_Ix != '-1':
                                            gpuId_child_2 = comm_info[commId]['rank_To_rankInfo'][child_2_Ix]['gpuId']
                                            goal_rank_child_2 = comm_info[commId]['rank_To_rankInfo'][child_2_Ix]['goal_rank']

                                            my_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_2_Ix]
                                            child_2_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank_child_2][gpuId_child_2][commId][event['event_type']][event['seq']][channel_id]['send'][myIx]

                                            if goal_rank_child_2 == goal_rank:
                                                for i in range(len(my_event_task_counter)):
                                                    my_receive_task_counter = my_event_task_counter[i]
                                                    child_2_send_task_counter = child_2_event_task_counter[i]
                                                    file.write(f"l{int(my_receive_task_counter)} requires l{int(child_2_send_task_counter)}\n")
                                        
                                        child_3_Ix = channel_info[channel_id]['child_3_rank']  ## local rank index in the communicator
                                        if child_3_Ix != '-1':
                                            gpuId_child_3 = comm_info[commId]['rank_To_rankInfo'][child_3_Ix]['gpuId']
                                            goal_rank_child_3 = comm_info[commId]['rank_To_rankInfo'][child_3_Ix]['goal_rank']

                                            my_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_3_Ix]
                                            child_3_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank_child_3][gpuId_child_3][commId][event['event_type']][event['seq']][channel_id]['send'][myIx]

                                            if goal_rank_child_3 == goal_rank:
                                                for i in range(len(my_event_task_counter)):
                                                    my_receive_task_counter = my_event_task_counter[i]
                                                    child_3_send_task_counter = child_3_event_task_counter[i]
                                                    file.write(f"l{int(my_receive_task_counter)} requires l{int(child_3_send_task_counter)}\n")
                                        
                                        parent_Ix = channel_info[channel_id]['parent_rank']  ## local rank index in the communicator
                                        if parent_Ix != '-1':
                                            gpuId_parent = comm_info[commId]['rank_To_rankInfo'][parent_Ix]['gpuId']
                                            goal_rank_parent = comm_info[commId]['rank_To_rankInfo'][parent_Ix]['goal_rank']

                                            my_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix]
                                            parent_event_task_counter = SendRecvEvents_To_TaskCounter[goal_rank_parent][gpuId_parent][commId][event['event_type']][event['seq']][channel_id]['send'][myIx]

                                            if goal_rank_parent == goal_rank:
                                                for i in range(len(my_event_task_counter)):
                                                    my_receive_task_counter = my_event_task_counter[i]
                                                    parent_send_task_counter = parent_event_task_counter[i]
                                                    file.write(f"l{int(my_receive_task_counter)} requires l{int(parent_send_task_counter)}\n")

                            elif event['event_type'] == 'Recv':  ## Intra-node Recv requires Send
                                commId = event['commId']

                                my_Ix = comm_info[commId]['gpuId_To_rank'][gpuId]
                                p2p_event_type = event['event_type']
                                p2p_peer_Ix = event['peer_rank']
                                gpuId_peer = comm_info[commId]['rank_To_rankInfo'][p2p_peer_Ix]['gpuId']
                                goal_rank_peer = comm_info[commId]['rank_To_rankInfo'][p2p_peer_Ix]['goal_rank']
                                p2p_seq = event['seq']
                                    
                                channel_id = 0

                                proto = event['protocol']
                                chunkSize = event['chunkSize']
                                count = event['count']

                                # if proto == '0': ## LL
                                #     chunkSize //= 2
                                #     for elemOffset in range(0, count, chunkSize):
                                #         nelem = int(min(chunkSize, count - elemOffset))
                                #         nelem = 0 if nelem < 0 else nelem

                                #         task_counter += 1
                                #         tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                #         if p2p_event['event_type'] == 'Send':
                                #             file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {p2p_event['peer_rank']}\n")
                                #         elif p2p_event['event_type'] == 'Recv':
                                #             file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {p2p_event['peer_rank']}\n")
                                #         file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                                #         file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")

                                if proto == '2': ## Simple
                                    recv_calc_task_counter = SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId]['Recv'][p2p_peer_Ix][p2p_seq][channel_id]
                                    send_calc_task_counter = SendRecvEvents_To_TaskCounter[goal_rank_peer][gpuId_peer][commId]['Send'][my_Ix][p2p_seq][channel_id]

                                    if goal_rank_peer == goal_rank and gpuId_peer != gpuId:
                                        for i in range(len(recv_calc_task_counter)):
                                            my_receive_task_counter = recv_calc_task_counter[i]
                                            peer_send_task_counter = send_calc_task_counter[i]
                                            file.write(f"l{int(my_receive_task_counter)} requires l{int(peer_send_task_counter)}\n")                           

            file.write("}\n")
        
        print(f"[DEBUG] Max comp time: {max_comp_time} ns ({max_comp_time / 1e9:.5f} s)")
