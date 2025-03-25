import argparse
import yaml
import os
import json
import math
import sqlite3
import re
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from queue import Queue

#### Postprocessing nsys files
def get_nsys_events(dir_path):
    comm_info = {}
    nccl_events = {}
    profile_interval = {}
    cupti_kernel_results = {}
    HostName_To_GoalRank = {}
    GoalRank_To_NumOfGPUs = {}
    commHash_to_commId = {}
    stream_to_streamId = {}
    comm_init_events = {}
    events_counter = {}
    ts_group_start= {}
    ts_group_end = {}
    gpuId = -1
    known_gpus = -1
    file_count = 0
    for file_name in os.listdir(dir_path):  ## each file may represent a host(root process), containing info of all GPUs (one GPU per child process) or a process corresponding to one GPU
        if file_name.endswith('.sqlite'):
            file_count += 1
            file_path = os.path.join(dir_path, file_name)
            
            pid_to_gpuId = {}

            Parse_State = {}
            last_Coll_streamId = {}
            last_P2P_streamId = {}
            last_update = {}

            pattern_HostName = r'nsys_report_([^.]+)\.'

            match = re.search(pattern_HostName, file_name)
            if match:
                host_name = match.group(1)
                print(f'Host {file_count} Name: {host_name}')

            if host_name in HostName_To_GoalRank:
                goal_rank = HostName_To_GoalRank[host_name]
                GoalRank_To_NumOfGPUs[goal_rank] += 1
            else:
                goal_rank = len(HostName_To_GoalRank)
                HostName_To_GoalRank[host_name] = goal_rank
                GoalRank_To_NumOfGPUs[goal_rank] = 1
                nccl_events[goal_rank] = {}
                cupti_kernel_results[goal_rank] = {}
                comm_init_events[goal_rank] = {}
                events_counter[goal_rank] = {}

            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            
            pattern_nsys_profile_start = r"nsys profiling start, pid: (\d+)"
            pattern_nsys_profile_end = r"nsys profiling stopped, pid: (\d+)"

            pattern_Comm_Info = r'commHash (\S+) commId (\S+) rank (\d+) nranks (\d+) pid (\d+)'
            pattern_Comm_NumOfChannels = r'(\d+) coll channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer, pid (\d+)'

            pattern_Ring = r'commHash (\S+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)'
            pattern_Tree = r'commHash (\S+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)'

            # Find communicator info
            cursor.execute("SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'commHash%'")
            nvtx_events_results = cursor.fetchall()

            for row in nvtx_events_results:
                if row[0]:
                    match_Comm_Info = re.search(pattern_Comm_Info, row[0])
                    match_Comm_NumOfChannels = re.search(pattern_Comm_NumOfChannels, row[0])

                    match_Ring = re.search(pattern_Ring, row[0])
                    match_Tree = re.search(pattern_Tree, row[0])

                    if match_Comm_Info:  # 'commHash (\S+) commId (\S+) rank (\d+) nranks (\d+) pid (\d+)'
                        commHash = match_Comm_Info.group(1)
                        commId = match_Comm_Info.group(2)
                        my_rank = match_Comm_Info.group(3)
                        nranks = match_Comm_Info.group(4)
                        pid = match_Comm_Info.group(5)

                        ts_init_start = row[1] ## ns
                        ts_init_end = row[2] ## ns

                        if commId not in comm_info:
                            comm_info[commId] = {}
                            comm_info[commId]['nranks'] = int(nranks)
                            comm_info[commId]['gpuId_To_rank'] = {}
                            comm_info[commId]['rank_To_rankInfo'] = {}
                            comm_info[commId]['comm_index'] = len(comm_info) - 1

                        if pid not in pid_to_gpuId:
                            known_gpus += 1
                            gpuId = known_gpus
                            pid_to_gpuId[pid] = gpuId
                            commHash_to_commId[gpuId] = {}
                            stream_to_streamId[gpuId] = {}
                            Parse_State[gpuId] = 0  ## awaiting P2P or Group operations
                            nccl_events[goal_rank][gpuId] = {}    
                            cupti_kernel_results[goal_rank][gpuId] = {}
                            events_counter[goal_rank][gpuId] = {}

                        gpuId = pid_to_gpuId[pid]
                        comm_info[commId]['gpuId_To_rank'][gpuId] = my_rank
                        comm_info[commId]['rank_To_rankInfo'][my_rank] = {
                            'gpuId': gpuId,
                            'goal_rank': goal_rank,
                            'host_name': host_name,
                            'channel_info': {
                                'Ring': [],
                                'Tree': []
                            }
                        }

                        commHash_to_commId[gpuId][commHash] = commId
                        last_commId = commId

                        if gpuId not in comm_init_events[goal_rank]:
                            comm_init_events[goal_rank][gpuId] = {}
                            comm_init_events[goal_rank][gpuId]['ts_init_start'] = ts_init_start
                            comm_init_events[goal_rank][gpuId]['ts_init_end'] = ts_init_end

                    elif match_Comm_NumOfChannels:
                        num_P2P_channels_per_peer = match_Comm_NumOfChannels.group(4)
                        comm_info[last_commId]['NumOfP2PChannelsPerPeer'] = num_P2P_channels_per_peer

                    elif match_Ring:  ## 'commHash (\S+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)'
                        commHash = match_Ring.group(1)
                        channel_Id = match_Ring.group(2)
                        previous_rank = match_Ring.group(3)
                        my_rank = match_Ring.group(4)
                        next_rank = match_Ring.group(5)
                        pid = match_Ring.group(6)

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        comm_info[commId]['rank_To_rankInfo'][my_rank]['channel_info']['Ring'].append(
                            {
                                'previous_rank': previous_rank,
                                'my_rank': my_rank,
                                'next_rank': next_rank,
                                'channel_Id': channel_Id
                            }
                        )

                    elif match_Tree:  ## 'commHash (\S+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)'
                        commHash = match_Tree.group(1)
                        channel_Id = match_Tree.group(2)
                        child_1_rank = match_Tree.group(3)
                        child_2_rank = match_Tree.group(4)
                        child_3_rank = match_Tree.group(5)
                        my_rank = match_Tree.group(6)
                        parent_rank = match_Tree.group(7)
                        pid = match_Tree.group(8)

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        comm_info[commId]['rank_To_rankInfo'][my_rank]['channel_info']['Tree'].append(
                            {
                                'child_1_rank': child_1_rank,
                                'child_2_rank': child_2_rank,
                                'child_3_rank': child_3_rank,
                                'my_rank': my_rank,
                                'parent_rank': parent_rank,
                                'channel_Id': channel_Id
                            }
                        )
            
            # Fetch information about profiling interval
            cursor.execute("SELECT text, start FROM NVTX_EVENTS WHERE text LIKE 'nsys profiling%'")
            nvtx_events_results = cursor.fetchall()

            for row in nvtx_events_results:
                if row[0]:
                    match_profile_start = re.search(pattern_nsys_profile_start, row[0])
                    match_profile_end = re.search(pattern_nsys_profile_end, row[0])

                    if match_profile_start:
                        pid = match_profile_start.group(1)
                        assert pid in pid_to_gpuId, f'[ERROR] pid {pid} not in pid_to_gpuId'
                        gpuId = pid_to_gpuId[pid]
                        ts_start = row[1] ## ns
                        assert gpuId not in profile_interval, f'[ERROR] gpuId {gpuId} already in profile_interval'
                        
                        profile_interval[gpuId] = {}
                        profile_interval[gpuId]["start"] = ts_start
                    
                    elif match_profile_end:
                        pid = match_profile_end.group(1)
                        assert pid in pid_to_gpuId, f'[ERROR] pid {pid} not in pid_to_gpuId'
                        gpuId = pid_to_gpuId[pid]
                        ts_end = row[1]
                        assert gpuId in profile_interval, f'[ERROR] gpuId {gpuId} not in profile_interval'
                        profile_interval[gpuId]["end"] = ts_end

            cursor.execute('SELECT text, start, end FROM NVTX_EVENTS')  ## row[0]: text, row[1]: ts_start, row[2]: ts_end
            nvtx_events_results = cursor.fetchall()

            pattern_nccl_AllReduce = r'ncclAllReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'
            pattern_nccl_Broadcast = r'ncclBroadcast\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), root (\d+), pid (\d+)'
            pattern_nccl_Reduce = r'ncclReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), root (\d+), pid (\d+)'
            pattern_nccl_AllGather = r'ncclAllGather\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), pid (\d+)'
            pattern_nccl_ReduceScatter = r'ncclReduceScatter\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'

            pattern_nccl_Send = r'ncclSend\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), receiver_rank (\d+), pid (\d+)'
            pattern_nccl_Recv = r'ncclRecv\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), sender_rank (\d+), pid (\d+)'

            pattern_nccl_GroupStart = r'ncclGroupStart\(\): pid (\d+)'
            pattern_nccl_GroupEnd = r'ncclGroupEnd\(\): pid (\d+)'

            pattern_Coll_Info = r'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
            pattern_Coll_Elem = r'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)'

            pattern_P2P_Elem = r'Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)'

            pattern_ncclKernel = r'ncclLaunchKernel\(\): pid (\d+)'

            pid_pattern = r'pid (\d+)'

            for row in tqdm(nvtx_events_results):
                if row[0]:
                    # If profiling interval is found, ignore all events outside of it
                    match_pid = re.search(pid_pattern, row[0])
                    if match_pid:
                        pid = match_pid.group(1)
                        assert pid in pid_to_gpuId, f'[ERROR] pid {pid} not in pid_to_gpuId'
                        gpuId = pid_to_gpuId[pid]
                        # if gpuId not in profile_interval:
                        #     continue
                        if (gpuId in profiling_interval) and \
                            (row[1] < profile_interval[gpuId]["start"] or row[1] > profile_interval[gpuId]["end"]):
                            continue

                    match_profile_start = re.search(pattern_nsys_profile_start, row[0])
                    match_profile_end = re.search(pattern_nsys_profile_end, row[0])

                    match_Comm_Info = re.search(pattern_Comm_Info, row[0])
                    match_Comm_NumOfChannels = re.search(pattern_Comm_NumOfChannels, row[0])

                    match_Ring = re.search(pattern_Ring, row[0])
                    match_Tree = re.search(pattern_Tree, row[0])

                    match_nccl_AllReduce = re.search(pattern_nccl_AllReduce, row[0])
                    match_nccl_Broadcast = re.search(pattern_nccl_Broadcast, row[0])
                    match_nccl_Reduce = re.search(pattern_nccl_Reduce, row[0])
                    match_nccl_AllGather = re.search(pattern_nccl_AllGather, row[0])
                    match_nccl_ReduceScatter = re.search(pattern_nccl_ReduceScatter, row[0])

                    match_nccl_Send = re.search(pattern_nccl_Send, row[0])
                    match_nccl_Recv = re.search(pattern_nccl_Recv, row[0])

                    match_nccl_GroupStart = re.search(pattern_nccl_GroupStart, row[0])
                    match_nccl_GroupEnd = re.search(pattern_nccl_GroupEnd, row[0])

                    match_Coll_Info = re.search(pattern_Coll_Info, row[0])
                    match_Coll_Elem = re.search(pattern_Coll_Elem, row[0])    

                    match_P2P_Elem = re.search(pattern_P2P_Elem, row[0])

                    match_ncclLaunchKernel = re.search(pattern_ncclKernel, row[0])


                    if match_nccl_AllReduce:  ## 'ncclAllReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'
                        commHash = match_nccl_AllReduce.group(1)
                        stream = match_nccl_AllReduce.group(2)
                        data_size = int(match_nccl_AllReduce.group(3))
                        type_size = int(match_nccl_AllReduce.group(4))
                        red_op = match_nccl_AllReduce.group(5)
                        pid = match_nccl_AllReduce.group(6)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllReduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllReduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'AllReduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'red_op': red_op,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['AllReduce']
                                    }
                                )    
                                
                                events_counter[goal_rank][gpuId][commId]['AllReduce'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllReduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllReduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'AllReduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_Broadcast:  ## 'ncclBroadcast\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), root (\d+)'
                        commHash = match_nccl_Broadcast.group(1)
                        stream = match_nccl_Broadcast.group(2)
                        data_size = int(match_nccl_Broadcast.group(3))
                        type_size = int(match_nccl_Broadcast.group(4))
                        root_rank = match_nccl_Broadcast.group(5)
                        pid = match_nccl_Broadcast.group(6)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Broadcast' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Broadcast'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'Broadcast',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'root_rank': root_rank,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['Broadcast']
                                    }
                                ) 
                                
                                events_counter[goal_rank][gpuId][commId]['Broadcast'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Broadcast' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Broadcast'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'Broadcast',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_Reduce:  ## 'ncclReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), root (\d+), pid (\d+)'
                        commHash = match_nccl_Reduce.group(1)
                        stream = match_nccl_Reduce.group(2)
                        data_size = int(match_nccl_Reduce.group(3))
                        type_size = int(match_nccl_Reduce.group(4))
                        red_op = match_nccl_Reduce.group(5)
                        root_rank = match_nccl_Reduce.group(6)
                        pid = match_nccl_Reduce.group(7)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Reduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Reduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'Reduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'red_op': red_op,
                                        'root_rank': root_rank,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['Reduce']
                                    }
                                )    
                                
                                events_counter[goal_rank][gpuId][commId]['Reduce'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Reduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Reduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'Reduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_AllGather:  ## 'ncclAllGather\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), pid (\d+)'
                        commHash = match_nccl_AllGather.group(1)
                        stream = match_nccl_AllGather.group(2)
                        data_size = int(match_nccl_AllGather.group(3))
                        type_size = int(match_nccl_AllGather.group(4))
                        pid = match_nccl_AllGather.group(5)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns
                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllGather' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllGather'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'AllGather',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['AllGather']
                                    }
                                )
                                
                                events_counter[goal_rank][gpuId][commId]['AllGather'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllGather' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllGather'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'AllGather',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_ReduceScatter:  ## 'ncclReduceScatter\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+)'
                        commHash = match_nccl_ReduceScatter.group(1)
                        stream = match_nccl_ReduceScatter.group(2)
                        data_size = int(match_nccl_ReduceScatter.group(3))
                        type_size = int(match_nccl_ReduceScatter.group(4))
                        red_op = match_nccl_ReduceScatter.group(5)
                        pid = match_nccl_ReduceScatter.group(6)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'ReduceScatter' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['ReduceScatter'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'ReduceScatter',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'red_op': red_op,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['ReduceScatter']
                                    }
                                )
                                
                                events_counter[goal_rank][gpuId][commId]['ReduceScatter'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'ReduceScatter' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['ReduceScatter'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'ReduceScatter',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_Coll_Info: 
                        ## 'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
                        collType = int(match_Coll_Info.group(1))
                        root_rank = int(match_Coll_Info.group(2))
                        redOp = int(match_Coll_Info.group(3))
                        algo = match_Coll_Info.group(4)
                        proto = match_Coll_Info.group(5)
                        commHash = match_Coll_Info.group(6)
                        stream = match_Coll_Info.group(7)
                        data_size = int(match_Coll_Info.group(8))
                        type_size = int(match_Coll_Info.group(9))
                        
                        chunkSteps = int(match_Coll_Info.group(12))
                        sliceSteps = int(match_Coll_Info.group(13))
                        stepSize = int(match_Coll_Info.group(14)) 
                        pid = match_Coll_Info.group(15)

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        
                        if Parse_State[gpuId] == 0:
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['algorithm'] = algo
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['protocol'] = proto
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['chunkSteps'] = chunkSteps
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['sliceSteps'] = sliceSteps
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['stepSize'] = stepSize

                        elif Parse_State[gpuId] == 6:
                            CollType = nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_type']

                            if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                            if CollType not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId][CollType] = 0

                            assert commHash_to_commId[gpuId][commHash] == nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['commId'], 'not the same comm in groupoperation'
                            assert stream_to_streamId[gpuId][stream] == nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['streamId'], 'not the same stream in group operation 1'
                            assert stream_to_streamId[gpuId][stream] == last_Coll_streamId[gpuId], 'not the same stream in group operation 2'

                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'].append(
                                {
                                    'algorithm': algo,
                                    'protocol': proto,
                                    'data_size': data_size,
                                    'type_size': type_size,
                                    'root_rank': root_rank,
                                    'red_op': redOp,
                                    'seq': events_counter[goal_rank][gpuId][commId][CollType],
                                    'chunkSteps': chunkSteps,
                                    'sliceSteps': sliceSteps,
                                    'stepSize': stepSize
                                }
                            )

                            events_counter[goal_rank][gpuId][commId][CollType] += 1

                    elif match_Coll_Elem: ## 'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)'
                        nWarps = int(match_Coll_Elem.group(1))
                        count = int(match_Coll_Elem.group(2))
                        chunkCount = int(match_Coll_Elem.group(3))
                        workCount = int(match_Coll_Elem.group(4))
                        lastChunkCount = int(match_Coll_Elem.group(5))
                        workOffset = int(match_Coll_Elem.group(6))
                        sendbuff = int(match_Coll_Elem.group(7))
                        recvbuff = int(match_Coll_Elem.group(8))
                        pid = match_Coll_Elem.group(9)

                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 0:
                            if 'elems' not in nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]:
                                nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['elems'] = []

                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['elems'].append(
                                {
                                    'count': count,
                                    'chunkCount': chunkCount,
                                    'workCount': workCount,
                                    'lastChunkCount': lastChunkCount,
                                    'workOffset': workOffset,
                                    'sendbuff': sendbuff,
                                    'recvbuff': recvbuff,
                                }
                            )

                        elif Parse_State[gpuId] == 6:
                            if 'elems' not in nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'][-1]:
                                nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'][-1]['elems'] = []
                            
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'][-1]['elems'].append(
                                {
                                    'count': count,
                                    'chunkCount': chunkCount,
                                    'workCount': workCount,
                                    'lastChunkCount': lastChunkCount,
                                    'workOffset': workOffset,
                                    'sendbuff': sendbuff,
                                    'recvbuff': recvbuff,
                                }
                            )

                    elif match_nccl_GroupStart:
                        pid = match_nccl_GroupStart.group(1)

                        if pid not in pid_to_gpuId:
                            known_gpus += 1
                            gpuId = known_gpus
                            pid_to_gpuId[pid] = gpuId
                            commHash_to_commId[gpuId] = {}
                            stream_to_streamId[gpuId] = {}
                            Parse_State[gpuId] = 0  ## awaiting P2P or Group operations
                            nccl_events[goal_rank][gpuId] = {}    
                            cupti_kernel_results[goal_rank][gpuId] = {}
                            events_counter[goal_rank][gpuId] = {}

                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            ts_group_start[gpuId] = row[1] ## ns
                            Parse_State[gpuId] = 1  ## awaiting ncclColl or ncclP2P, ignore ncclGroupStart/ncclGroupEnd in between

                        elif Parse_State[gpuId] == 2:
                            Parse_State[gpuId] = 3

                        elif Parse_State[gpuId] == 7:
                            Parse_State[gpuId] = 8

                    elif match_nccl_GroupEnd:
                        pid = match_nccl_GroupEnd.group(1)
                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 3:
                            Parse_State[gpuId] = 2

                        elif Parse_State[gpuId] == 2:
                            ts_group_end[gpuId] = row[2] ## ns
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['ts_end'] = ts_group_end[gpuId]
                            Parse_State[gpuId] = 4

                        elif Parse_State[gpuId] == 5:
                            ts_group_end[gpuId] = row[2]  ## ns
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['ts_end'] = ts_group_end[gpuId]
                            Parse_State[gpuId] = 6

                        elif Parse_State[gpuId] == 1:  ## in case directly call ncclGroupEnd() after ncclGroupStart() 
                            Parse_State[gpuId] = 0

                        elif Parse_State[gpuId] == 8:
                            Parse_State[gpuId] = 7

                    elif match_nccl_Send:  ## 'ncclSend\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), receiver_rank (\d+), pid (\d+)'
                        commHash = match_nccl_Send.group(1)
                        stream = match_nccl_Send.group(2)
                        data_size = int(match_nccl_Send.group(3))
                        type_size = int(match_nccl_Send.group(4))
                        peer_rank = match_nccl_Send.group(5)
                        pid = match_nccl_Send.group(6)

                        gpuId = pid_to_gpuId[pid]

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0
                        
                        if Parse_State[gpuId] == 1:  ## Group P2P
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Send'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                    events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 2

                                last_P2P_streamId[gpuId] = streamId    
                                last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 2:  ## Group P2P
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]
                            streamId = stream_to_streamId[gpuId][stream]

                            if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId]['Send'] = {}

                            if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                            Parse_State[gpuId] = 2

                            last_P2P_streamId[gpuId] = streamId    
                            last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 0:  ## Single P2P
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Send'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                    events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 7

                                last_P2P_streamId[gpuId] = streamId    
                                last_update[gpuId] = 'P2P'

                    elif match_nccl_Recv:  ## 'ncclRecv\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), sender_rank (\d+)'
                        commHash = match_nccl_Recv.group(1)
                        stream = match_nccl_Recv.group(2)
                        data_size = int(match_nccl_Recv.group(3))
                        type_size = int(match_nccl_Recv.group(4))
                        peer_rank = match_nccl_Recv.group(5)
                        pid = match_nccl_Recv.group(6)

                        gpuId = pid_to_gpuId[pid]

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0
                        
                        if Parse_State[gpuId] == 1:  ## Group P2P
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                    events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 2

                                last_P2P_streamId[gpuId] = streamId
                                last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 2:  ## Group P2P
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]
                            streamId = stream_to_streamId[gpuId][stream]

                            if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                            if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                            Parse_State[gpuId] = 2

                            last_P2P_streamId[gpuId] = streamId
                            last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 0:  ## Single P2P
                            commId = commHash_to_commId[gpuId][commHash]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                    events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 7

                                last_P2P_streamId[gpuId] = streamId    
                                last_update[gpuId] = 'P2P'

                    elif match_P2P_Elem:  ## 'Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)'
                        p2pType = match_P2P_Elem.group(3)
                        peer_rank = match_P2P_Elem.group(4)
                        proto = match_P2P_Elem.group(5)
                        countHi32 = int(match_P2P_Elem.group(6))
                        countLo32 = int(match_P2P_Elem.group(7))
                        chunkSize = int(match_P2P_Elem.group(8))
                        pid  = match_P2P_Elem.group(9)

                        gpuId = pid_to_gpuId[pid]
                        commId = nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['commId']

                        if p2pType == '1':
                            p2p_type = 'Send' 
                        elif p2pType == '2':
                            p2p_type = 'Recv' 

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 7 or Parse_State[gpuId] == 9:
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['P2P_events'].append(
                                {
                                    'p2p_type': p2p_type,
                                    'peer_rank': peer_rank,
                                    'protocol': proto,
                                    'countHi32': countHi32,
                                    'countLo32': countLo32,
                                    'chunkSize': chunkSize,
                                    'count': countHi32 * 2**32 + countLo32,
                                    'seq': events_counter[goal_rank][gpuId][commId][p2p_type][peer_rank]
                                }
                            )

                            if Parse_State[gpuId] == 7:
                                Parse_State[gpuId] = 9

                            events_counter[goal_rank][gpuId][commId][p2p_type][peer_rank] += 1

                    elif match_ncclLaunchKernel:
                        pid = match_ncclLaunchKernel.group(1)

                        gpuId = pid_to_gpuId[pid]

                        ts_kernel = row[2] ## ns

                        if last_update[gpuId] == 'Coll':
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['ts_kernel'] = ts_kernel

                        elif last_update[gpuId] == 'P2P':
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['ts_kernel'] = ts_kernel
                
                    
            
            cursor.execute('SELECT globalPid, pid FROM PROCESSES')
            globalPid_pids = cursor.fetchall()
            pid_dict = {row[0]: row[1] for row in globalPid_pids}
            
            cursor.execute('SELECT id, value FROM StringIds')
            string_ids = cursor.fetchall()
            string_dict = {row[0]: row[1] for row in string_ids}
            
            cursor.execute('SELECT start, end, streamId, globalPid, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL')
            cupti_kernel_events = cursor.fetchall()
            for row in cupti_kernel_events:
                start, end, streamId, globalPid, demangled_name = row
                if string_dict[demangled_name].startswith('ncclKernel') or string_dict[demangled_name].startswith('ncclDevKernel'):
                    fields = string_dict[demangled_name].replace('(', '_').replace(')', '_').split('_')
                    pid = pid_dict[globalPid]
                    assert str(pid) in pid_to_gpuId, f'pid {pid} not in pid_to_gpuId'
                    gpuId = pid_to_gpuId[str(pid)]

                    # Do not include the kernel if it is not inside the profile interval
                    if gpuId in profile_interval and \
                        (end < profile_interval[gpuId]['start'] or start > profile_interval[gpuId]['end']):
                        continue

                    if streamId not in cupti_kernel_results[goal_rank][gpuId]:
                        cupti_kernel_results[goal_rank][gpuId][streamId] = [] 

                    cupti_kernel_results[goal_rank][gpuId][streamId].append({
                        'gpu_event_type': fields[1],
                        'ts_gpu_start': start, ## ns
                        'ts_gpu_end': end, ## ns
                    })

            conn.close()

            for gpuId in nccl_events[goal_rank].keys():
                len_nccl_events = sum([len(v) for v in nccl_events[goal_rank][gpuId].values()])
                len_cupti_kernel_results = sum([len(v) for v in cupti_kernel_results[goal_rank][gpuId].values()])
                if len_nccl_events != len_cupti_kernel_results:
                    print(f'[ERROR] Host {goal_rank} gpu: {gpuId} Different number of events in nccl and cupti kernel results {len_nccl_events} != {len_cupti_kernel_results}')

                    # Dumps events to a file
                    with open(f'debug_nccl_events_{goal_rank}_{gpuId}.json', 'w') as f:
                        json.dump(nccl_events[goal_rank][gpuId], f, indent=4)
                    with open(f'debug_cupti_kernel_results_{goal_rank}_{gpuId}.json', 'w') as f:
                        json.dump(cupti_kernel_results[goal_rank][gpuId], f, indent=4)
                    exit(1)
    
    return comm_init_events, nccl_events, cupti_kernel_results, comm_info, HostName_To_GoalRank, profile_interval


def merge_stream_if_no_overlap(nccl_events, cupti_kernel_results):
    """
    Iterates through the NCCL events collected on all streams of a GPU and merges them if
    no overlap is detected.
    Returns the original nccl_events if no overlap is detected, otherwise returns the merged events.
    """
    merged_nccl_events = {}
    merged_cupti_kernel_results = {}

    for goal_rank, gpu_events in nccl_events.items():
        merged_nccl_events[goal_rank] = {}
        merged_cupti_kernel_results[goal_rank] = {}
        for gpuId in gpu_events.keys():
            merged_nccl_events[goal_rank][gpuId] = {}
            merged_cupti_kernel_results[goal_rank][gpuId] = {}
            tmp_nccl_events = []
            tmp_cupti_kernel_results = []

            for streamId, stream_events in nccl_events[goal_rank][gpuId].items():
                tmp_nccl_events.extend(stream_events)
            
            for streamId, stream_events in cupti_kernel_results[goal_rank][gpuId].items():
                tmp_cupti_kernel_results.extend(stream_events)
            
            tmp_nccl_events.sort(key=lambda x: x['ts_start'])
            tmp_cupti_kernel_results.sort(key=lambda x: x['ts_gpu_start'])
            assert len(tmp_nccl_events) == sum([len(v) for v in nccl_events[goal_rank][gpuId].values()]), 'Different number of events in nccl_events'
            assert len(tmp_cupti_kernel_results) == sum([len(v) for v in cupti_kernel_results[goal_rank][gpuId].values()]), 'Different number of events in cupti_kernel_results'

            # Checks if there is any overlap between the events
            assert len(tmp_nccl_events) == len(tmp_cupti_kernel_results), f'Different number of events in nccl and cupti kernel results {len(tmp_nccl_events)} != {len(tmp_cupti_kernel_results)}'
            for i in range(len(tmp_nccl_events) - 1):
                if tmp_nccl_events[i]['ts_end'] > tmp_nccl_events[i+1]['ts_start'] or \
                    tmp_cupti_kernel_results[i]['ts_gpu_end'] > tmp_cupti_kernel_results[i+1]['ts_gpu_start']:
                    print(f"[INFO] Overlap detected for GPU {gpuId} on goal rank {goal_rank}, not merging streams")
                    return nccl_events, cupti_kernel_results

            merged_nccl_events[goal_rank][gpuId][0] = tmp_nccl_events
            kernel_stream_id = list(cupti_kernel_results[goal_rank][gpuId].keys())[0]
            merged_cupti_kernel_results[goal_rank][gpuId][kernel_stream_id] = tmp_cupti_kernel_results
            print(f"[INFO] No overlap detected for GPU {gpuId} on goal rank {goal_rank}")
            print(f"[INFO] Streams from GPU {gpuId} have been merged into a single stream, number of events: {len(tmp_nccl_events)}")
    
    return merged_nccl_events, merged_cupti_kernel_results