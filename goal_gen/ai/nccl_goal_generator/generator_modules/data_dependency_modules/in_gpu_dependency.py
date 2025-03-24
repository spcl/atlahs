from .utils import modRanks, div_up, get_event_type
from .intra_node_gpu_transfer_time import get_intra_node_gpu_transfer_time
from .reduction_copy_time import get_reduction_time, get_copy_time
from tqdm import tqdm
from typing import Dict

def get_in_gpu_microevents_dependency(nccl_group_events, comm_init_events,
                                      comm_info, goal_file_name, profile_interval={},
                                      slient=True) -> Dict:
    
    num_ranks = len(nccl_group_events)
    # task_counter = 0
    SendRecvEvents_To_TaskCounter = {}
    # with open(goal_file_name, 'w') as file:
        # file.write(f"num_ranks {num_ranks}\n")

    for goal_rank in range(num_ranks):
        print(f"[DEBUG] goal_rank: {goal_rank}")
        task_counter = 0

        # file.write(f"\nrank {goal_rank}")
        # file.write(" {\n")

        SendRecvEvents_To_TaskCounter[goal_rank] = {}

        goal_events = nccl_group_events[goal_rank]
        task_counter += 1
        # file.write(f"l{task_counter}: calc 0\n") ## Start point of the node
        node_start_calc_id = task_counter
        
        task_counter += 1
        # file.write(f"l{task_counter}: calc 0\n") ## End point of the node
        node_end_calc_id = task_counter

        for gpuId, gpu_events in goal_events.items():
            SendRecvEvents_To_TaskCounter[goal_rank][gpuId] = {}

            gpu_all_stream_start_time = None
            if gpuId in profile_interval:
                # If nvtx mark is used
                gpu_all_stream_start_time = profile_interval[gpuId]["start"]
                gpu_all_stream_end_time = profile_interval[gpuId]["end"]
            else:
                for streamId, stream_events in gpu_events.items():
                    if gpu_all_stream_start_time is None:
                        gpu_all_stream_start_time = stream_events[0]['ts_group_gpu_start']
                    else:
                        gpu_all_stream_start_time = min(gpu_all_stream_start_time, stream_events[0]['ts_group_gpu_start'])
                gpu_all_stream_end_time = float('inf')
            
            print(f"[DEBUG] GPU {gpuId} profiling interval: [{gpu_all_stream_start_time}, {gpu_all_stream_end_time}]")

            for streamId, stream_events in tqdm(gpu_events.items()):
                last_group_event_end_time =  gpu_all_stream_start_time
                last_group_event_end_id = node_start_calc_id
                for group_event_index, group_event in enumerate(stream_events):
                    if group_event["ts_group_gpu_start"] < gpu_all_stream_start_time:
                        continue

                    if group_event["ts_group_gpu_start"] >= gpu_all_stream_end_time:
                        # file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")
                        break

                    launched = 0

                    task_counter += 1
                    # file.write(f"l{task_counter}: calc {group_event['ts_group_gpu_start'] - last_group_event_end_time}\n")  ## Former calc between first group host event start and last group gpu event end
                    # file.write(f"l{task_counter} requires l{last_group_event_end_id}\n")
                    group_event_start_calc_id = task_counter

                    task_counter += 1
                    # file.write(f"l{task_counter}: calc 0\n")  ## End calc of the parallel group of events
                    group_event_end_calc_id = task_counter
                    last_group_event_end_time = group_event['ts_group_gpu_end']
                    last_group_event_end_id = task_counter

                    for event in group_event['events']:
                        if event['event_type'] == 'Send' or event['event_type'] == 'Recv':
                            commId = event['commId']
                            p2p_event_type = event['event_type']
                            p2p_peer_Ix = event['peer_rank']
                            gpuId_peer = comm_info[commId]['rank_To_rankInfo'][p2p_peer_Ix]['gpuId']
                            p2p_seq = event['seq']

                            if commId not in SendRecvEvents_To_TaskCounter[goal_rank][gpuId]:
                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId] = {}

                            if launched == 0:
                                task_counter += 1
                                # file.write(f"l{task_counter}: calc 0\n")  ## Former calc between nccl kernel launch end and host event start
                                # file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                p2p_group_start_calc_id = task_counter

                                task_counter += 1
                                # file.write(f"l{task_counter}: calc 0\n")
                                # file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")
                                p2p_group_end_calc_id = task_counter

                                launched = 1

                            if p2p_event_type not in SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId]:
                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type] = {}  ## send or recv
                            
                            if p2p_peer_Ix not in SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type]:
                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type][p2p_peer_Ix] = {}
                            
                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type][p2p_peer_Ix][p2p_seq] = {}

                            channel_id = 0
                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type][p2p_peer_Ix][p2p_seq][channel_id] = []

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
                            #             # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {p2p_event['peer_rank']}\n")
                            #         elif p2p_event['event_type'] == 'Recv':
                            #             # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {p2p_event['peer_rank']}\n")
                            #         # file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                            #         # file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")

                            if proto == '2': ## Simple
                                for elemOffset in range(0, count, chunkSize):
                                    nelem = int(min(chunkSize, count - elemOffset))
                                    nelem = 0 if nelem < 0 else nelem

                                    task_counter += 1
                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type][p2p_peer_Ix][p2p_seq][channel_id])) + str(channel_id).zfill(2) + str(p2p_seq).zfill(4) + str(get_event_type(p2p_event_type)).zfill(1) + str(event['comm_index']).zfill(2)
                                    # if p2p_event_type == 'Send':
                                        # file.write(f"l{task_counter}: send {nelem}b to {p2p_peer_Ix} tag {tag}\n")
                                    # elif p2p_event_type == 'Recv':
                                        # file.write(f"l{task_counter}: recv {nelem}b from {p2p_peer_Ix} tag {tag}\n")
                                    # file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                                    # file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")

                                    if gpuId_peer != gpuId:
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type][p2p_peer_Ix][p2p_seq][channel_id].append(task_counter)
                                
                        else:
                            commId = event['commId']
                            nranks = comm_info[commId]['nranks']
                            if commId not in SendRecvEvents_To_TaskCounter[goal_rank][gpuId]:
                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId] = {}

                            if event['event_type'] not in SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId]:
                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']] = {}

                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']] = {}

                            if launched == 0:
                                task_counter += 1
                                # file.write(f"l{task_counter}: calc 0\n")  ## Former calc between nccl kernel launch end and host event start
                                # file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                gpu_event_start_calc_id = task_counter

                                task_counter += 1
                                # file.write(f"l{task_counter}: calc 0\n")  ## end calc of a gpu event
                                # file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")          
                                gpu_event_end_calc_id = task_counter     

                                launched = 1

                            if event['event_type'] == 'AllReduce':
                                algo = event['algorithm']  ## NCCL_ALGO_TREE: 0, NCCL_ALGO_RING: 1
                                proto = event['protocol']  ## NCCL_PROTO_LL: 0, NCCL_PROTO_LL128: 1, NCCL_PROTO_SIMPLE: 2
                                type_size = event['type_size']
                                chunkSteps = event['chunkSteps']
                                sliceSteps = event['sliceSteps']
                                stepSize = event['stepSize']

                                if algo == '1': ## Ring AllReduce
                                    ringIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][ringIx]['channel_info']['Ring']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id] = {}
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'] = {}
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'] = {}
                                        nranks = comm_info[event['commId']]['nranks']  ## 2
                                        prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator  ## potentially some allreduce use more elems than channels, maybe modify channel_id to 0
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx] = []
                                        nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx] = []
                                        
                                        chunkCount = elem['chunkCount']
                                        gridOffset = elem['workOffset']
                                        channelCount = elem['workCount']
                                        lastChunkCount = elem['lastChunkCount']
                                        loopCount = nranks * chunkCount

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
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    # file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                                    # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            task_counter += 1
                                                            # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                            # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                        
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
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    # file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto)}\n")
                                                    # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            task_counter += 1
                                                            # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto)}\n")
                                                            # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                            
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
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break

                                elif algo == '0': ## Tree AllReduce
                                    myIx = comm_info[commId]['gpuId_To_rank'][gpuId]  ## local rank index in the communicator
                                    channel_info = comm_info[commId]['rank_To_rankInfo'][myIx]['channel_info']['Tree']

                                    elems = event['elems']
                                    for channel_id, elem in enumerate(elems):
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id] = {}
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'] = {}
                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'] = {}
                                        nranks = comm_info[event['commId']]['nranks']  ## 2
                                        child_1_Ix = channel_info[channel_id]['child_1_rank']  ## local rank index in the communicator
                                        if child_1_Ix != '-1':
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_1_Ix] = []
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_1_Ix] = []
                                        child_2_Ix = channel_info[channel_id]['child_2_rank']  ## local rank index in the communicator
                                        if child_2_Ix != '-1':
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_2_Ix] = []
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_2_Ix] = []
                                        child_3_Ix = channel_info[channel_id]['child_3_rank']  ## local rank index in the communicator
                                        if child_3_Ix != '-1':
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_3_Ix] = []
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_3_Ix] = []
                                        parent_Ix = channel_info[channel_id]['parent_rank']  ## local rank index in the communicator
                                        if parent_Ix != '-1':
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix] = []
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix] = []
                                        
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
                                                    # file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)}\n")
                                                    calc_task_id = task_counter

                                                    for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                        if child_Ix != '-1':
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {child_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            # file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)

                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {child_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix].append(task_counter)

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
                                                            # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)}\n")
                                                            calc_task_id = task_counter

                                                            for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                if child_Ix != '-1':
                                                                    task_counter += 1
                                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {child_Ix} tag {tag}\n")
                                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    # file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)

                                                                    task_counter += 1
                                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {child_Ix} tag {tag}\n")
                                                                    # file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix].append(task_counter)
                                                            
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
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {parent_Ix} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                    task_counter += 1  ## Recv
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {parent_Ix} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix].append(task_counter)

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
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {parent_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                            task_counter += 1  ## Recv
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {parent_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix].append(task_counter)

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
                                                    # file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                                    calc_task_id = task_counter

                                                    for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                        if child_Ix != '-1':
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {child_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            # file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)
                                
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {parent_Ix} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                    ## RecvCopySend
                                                    task_counter += 1
                                                    # file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto)}\n")
                                                    calc_task_id = task_counter

                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {parent_Ix} tag {tag}\n")
                                                    # file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix].append(task_counter)
                                                    
                                                    for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                        if child_Ix != '-1':
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {child_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix].append(task_counter)

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
                                                            # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                            calc_task_id = task_counter

                                                            for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                if child_Ix != '-1':
                                                                    task_counter += 1
                                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {child_Ix} tag {tag}\n")
                                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    # file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)
                                        
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {parent_Ix} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                            ## RecvCopySend
                                                            task_counter += 1
                                                            # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto)}\n")
                                                            calc_task_id = task_counter

                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {parent_Ix} tag {tag}\n")
                                                            # file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix].append(task_counter)
                                                            
                                                            for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                if child_Ix != '-1':
                                                                    task_counter += 1
                                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {child_Ix} tag {tag}\n")
                                                                    # file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix].append(task_counter)
                                                            
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
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'] = {}
                                    nranks = comm_info[event['commId']]['nranks']  ## 2
                                    prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx] = []
                                    nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx] = []
                                    
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

                                        if (ringIx == root_rank): 
                                            if proto == '0':
                                                # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                if sendbuff == recvbuff:  ## In-Place: Send
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                
                                                else:  ## CopySend
                                                    task_counter += 1
                                                    # file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                                        else:  ## CopySend
                                                            task_counter += 1
                                                            # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                            # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)


                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break

                                        elif nextIx == root_rank: ## Recv
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break
                                            
                                        else:  ## RecvCopySend
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                        
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
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'] = {}
                                    nranks = comm_info[event['commId']]['nranks']
                                    prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx] = []
                                    nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx] = []
                                    
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
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                            else:  ## CopySend
                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                                    else:  ## CopySend
                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                                    slice += 1
                                                    offset += sliceSize

                                                    if not (slice < SlicePerChunk and offset < nelem):
                                                        break
                                                        
                                        ## Step 1 to step (k - 2): RecvCopySend
                                        for j in range(1, nranks - 1):
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                        
                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break

                                        ## Step (k - 1): Recv
                                        if proto == '0':
                                            task_counter += 1
                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                            # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

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
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

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
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'] = {}
                                    nranks = comm_info[event['commId']]['nranks']
                                    prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx] = []
                                    nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx] = []
                                    
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
                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                            # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                                    slice += 1
                                                    offset += sliceSize

                                                    if not (slice < SlicePerChunk and offset < nelem):
                                                        break
                                                        
                                        ## Step 1 to step (k - 2): RecvReduceSend
                                        for j in range(1, nranks - 1):
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                        
                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break

                                        ## Step (k - 1): RecvReduceCopy
                                        if proto == '0':
                                            task_counter += 1
                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                            # file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                            # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                            task_counter += 1
                                            # file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                            # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                            # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

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
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    # file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                    # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                    # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

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
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'] = {}
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'] = {}
                                    nranks = comm_info[event['commId']]['nranks']  ## 2
                                    prevIx = channel_info[channel_id]['previous_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx] = []
                                    nextIx = channel_info[channel_id]['next_rank']  ## local rank index in the communicator
                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx] = []
                                    
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

                                        if (prevIx == root_rank):  # Send
                                            if proto == '0':
                                                # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break

                                        elif ringIx == root_rank: ## RecvReduceCopy
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_reduction_time(nelem, proto) + get_copy_time(nelem, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize, proto) + get_copy_time(sliceSize, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        
                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break
                                            
                                        else:  ## RecvReduceSend
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                # file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                # file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        # file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        # file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        # file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                        
                                                        slice += 1
                                                        offset += sliceSize

                                                        if not (slice < SlicePerChunk and offset < nelem):
                                                            break   

                            else:
                                task_counter += 1
                                # file.write(f"l{task_counter}: {event['event_type']} {event['data_size']} bytes comm {event['comm_index']} gpu {gpuId} stream {streamId}\n")  ## gpu event
                                # file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                # file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")  
                    
                    # if group_event_index == len(stream_events) - 1:
                        # file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")

            # file.write("}\n")

    return SendRecvEvents_To_TaskCounter
