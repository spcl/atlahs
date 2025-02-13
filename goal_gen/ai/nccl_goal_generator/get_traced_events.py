import argparse
import yaml
import os
import json
import math
import sqlite3
import re
import numpy as np
import random
from scipy import interpolate
import importlib.resources
from collections import defaultdict
from queue import Queue

#### Postprocessing nsys files
def get_nsys_events(dir_path):
    comm_info = {}
    nccl_events = {}
    cupti_kernel_results = {}
    HostName_To_GoalRank = {}
    GoalRank_To_NumOfGPUs = {}
    comm_to_commId = {}
    stream_to_streamId = {}
    comm_init_events = {}
    events_counter = {}
    ts_group_start= {}
    ts_group_end = {}
    gpuId = -1
    known_gpus = -1

    for file_name in os.listdir(dir_path):  ## each file may represent a host(root process), containing info of all GPUs (one GPU per child process) or a process corresponding to one GPU
        if file_name.endswith('.sqlite'):
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
                print(f'Host Name: {host_name}')

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
            cursor.execute('SELECT text, start, end FROM NVTX_EVENTS')  ## row[0]: text, row[1]: ts_start, row[2]: ts_end
            nvtx_events_results = cursor.fetchall()

            pattern_Comm_Info = r'comm (\S+) commId (\S+) rank (\d+) nranks (\d+) pid (\d+)'
            pattern_Comm_NumOfChannels = r'(\d+) coll channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer, pid (\d+)'

            pattern_Ring = r'comm (\S+) commHash (\S+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)'
            pattern_Tree = r'comm (\S+) commHash (\S+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)'

            pattern_nccl_AllReduce = r'ncclAllReduce\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'
            pattern_nccl_Broadcast = r'ncclBroadcast\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), root (\d+), pid (\d+)'
            pattern_nccl_AllGather = r'ncclAllGather\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), pid (\d+)'
            pattern_nccl_ReduceScatter = r'ncclReduceScatter\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'

            pattern_nccl_Send = r'ncclSend\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), receiver_rank (\d+), pid (\d+)'
            pattern_nccl_Recv = r'ncclRecv\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), sender_rank (\d+), pid (\d+)'

            pattern_nccl_GroupStart = r'ncclGroupStart\(\): pid (\d+)'
            pattern_nccl_GroupEnd = r'ncclGroupEnd\(\): pid (\d+)'

            pattern_Coll_Info = r'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) comm (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
            pattern_Coll_Elem = r'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) pid (\d+)'

            pattern_P2P_Elem = r'Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)'

            pattern_ncclKernel = r'ncclLaunchKernel\(\): pid (\d+)'

            for row in nvtx_events_results:
                if row[0]:
                    match_Comm_Info = re.search(pattern_Comm_Info, row[0])
                    match_Comm_NumOfChannels = re.search(pattern_Comm_NumOfChannels, row[0])

                    match_Ring = re.search(pattern_Ring, row[0])
                    match_Tree = re.search(pattern_Tree, row[0])

                    match_nccl_AllReduce = re.search(pattern_nccl_AllReduce, row[0])
                    match_nccl_Broadcast = re.search(pattern_nccl_Broadcast, row[0])
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

                    if match_Comm_Info:
                        comm = match_Comm_Info.group(1)
                        commId = match_Comm_Info.group(2)
                        my_rank = match_Comm_Info.group(3)
                        nranks = match_Comm_Info.group(4)
                        pid = match_Comm_Info.group(5)

                        ts_init_start = row[1] // 1000  ## ns to us
                        ts_init_end = row[2] // 1000  ## ns to us

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
                            comm_to_commId[gpuId] = {}
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

                        comm_to_commId[gpuId][comm] = commId
                        last_commId = commId

                        if gpuId not in comm_init_events[goal_rank]:
                            comm_init_events[goal_rank][gpuId] = {}
                            comm_init_events[goal_rank][gpuId]['ts_init_start'] = ts_init_start
                            comm_init_events[goal_rank][gpuId]['ts_init_end'] = ts_init_end

                    elif match_Comm_NumOfChannels:
                        num_P2P_channels_per_peer = match_Comm_NumOfChannels.group(4)
                        comm_info[last_commId]['NumOfP2PChannelsPerPeer'] = num_P2P_channels_per_peer

                    elif match_Ring:
                        comm = match_Ring.group(1)
                        channel_Id = match_Ring.group(3)
                        previous_rank = match_Ring.group(4)
                        my_rank = match_Ring.group(5)
                        next_rank = match_Ring.group(6)
                        pid = match_Ring.group(7)

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
                        comm_info[commId]['rank_To_rankInfo'][my_rank]['channel_info']['Ring'].append(
                            {
                                'previous_rank': previous_rank,
                                'my_rank': my_rank,
                                'next_rank': next_rank,
                                'channel_Id': channel_Id
                            }
                        )

                    elif match_Tree:
                        comm = match_Tree.group(1)
                        channel_Id = match_Tree.group(3)
                        child_1_rank = match_Tree.group(4)
                        child_2_rank = match_Tree.group(5)
                        child_3_rank = match_Tree.group(6)
                        my_rank = match_Tree.group(7)
                        parent_rank = match_Tree.group(8)
                        pid = match_Tree.group(9)

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
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

                    elif match_nccl_AllReduce:  ## 'ncclAllReduce\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'
                        comm = match_nccl_AllReduce.group(1)
                        stream = match_nccl_AllReduce.group(2)
                        data_size = int(match_nccl_AllReduce.group(3))
                        type_size = int(match_nccl_AllReduce.group(4))
                        red_op = match_nccl_AllReduce.group(5)
                        pid = match_nccl_AllReduce.group(6)

                        ts_start = row[1] // 1000  ## ns to us
                        ts_end = row[2] // 1000  ## ns to us

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1:
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
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1:
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

                    elif match_nccl_Broadcast:  ## 'ncclBroadcast\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), root (\d+)'
                        comm = match_nccl_Broadcast.group(1)
                        stream = match_nccl_Broadcast.group(2)
                        data_size = int(match_nccl_Broadcast.group(3))
                        type_size = int(match_nccl_Broadcast.group(4))
                        root_rank = match_nccl_Broadcast.group(5)
                        pid = match_nccl_Broadcast.group(6)

                        ts_start = row[1] // 1000  ## ns to us
                        ts_end = row[2] // 1000  ## ns to us

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1:
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
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1:
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

                    elif match_nccl_AllGather:  ## 'ncclAllGather\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), pid (\d+)'
                        comm = match_nccl_AllGather.group(1)
                        stream = match_nccl_AllGather.group(2)
                        data_size = int(match_nccl_AllGather.group(3))
                        type_size = int(match_nccl_AllGather.group(4))
                        pid = match_nccl_AllGather.group(5)

                        ts_start = row[1] // 1000  ## ns to us
                        ts_end = row[2] // 1000  ## ns to us

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1:
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
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1:
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

                    elif match_nccl_ReduceScatter:  ## 'ncclReduceScatter\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+)'
                        comm = match_nccl_ReduceScatter.group(1)
                        stream = match_nccl_ReduceScatter.group(2)
                        data_size = int(match_nccl_ReduceScatter.group(3))
                        type_size = int(match_nccl_ReduceScatter.group(4))
                        red_op = match_nccl_ReduceScatter.group(5)
                        pid = match_nccl_ReduceScatter.group(6)

                        ts_start = row[1] // 1000  ## ns to us
                        ts_end = row[2] // 1000  ## ns to us

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
                        my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1:
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
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1:
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
                        ## 'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) comm (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
                        collType = int(match_Coll_Info.group(1))
                        root = int(match_Coll_Info.group(2))
                        redOp = int(match_Coll_Info.group(3))
                        algo = match_Coll_Info.group(4)
                        proto = match_Coll_Info.group(5)
                        comm = match_Coll_Info.group(6)
                        stream = match_Coll_Info.group(7)
                        data_size = int(match_Coll_Info.group(8))
                        type_size = int(match_Coll_Info.group(9))
                        
                        chunkSteps = int(match_Coll_Info.group(12))
                        sliceSteps = int(match_Coll_Info.group(13))
                        stepSize = int(match_Coll_Info.group(14)) 
                        pid = match_Coll_Info.group(15)

                        gpuId = pid_to_gpuId[pid]
                        commId = comm_to_commId[gpuId][comm]
                        
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

                            assert comm_to_commId[gpuId][comm] == nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['commId'], 'not the same comm in groupoperation'
                            assert stream_to_streamId[gpuId][stream] == nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['streamId'], 'not the same stream in group operation 1'
                            assert stream_to_streamId[gpuId][stream] == last_Coll_streamId[gpuId], 'not the same stream in group operation 2'

                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'].append(
                                {
                                    'algorithm': algo,
                                    'protocol': proto,
                                    'data_size': data_size,
                                    'type_size': type_size,
                                    'root': root,
                                    'red_op': redOp,
                                    'seq': events_counter[goal_rank][gpuId][commId][CollType],
                                    'chunkSteps': chunkSteps,
                                    'sliceSteps': sliceSteps,
                                    'stepSize': stepSize
                                }
                            )

                            events_counter[goal_rank][gpuId][commId][CollType] += 1

                    elif match_Coll_Elem: ## 'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+)'
                        nWarps = int(match_Coll_Elem.group(1))
                        count = int(match_Coll_Elem.group(2))
                        chunkCount = int(match_Coll_Elem.group(3))
                        workCount = int(match_Coll_Elem.group(4))
                        lastChunkCount = int(match_Coll_Elem.group(5))
                        workOffset = int(match_Coll_Elem.group(6))
                        pid = match_Coll_Elem.group(7)

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
                                }
                            )

                    elif match_nccl_GroupStart:
                        pid = match_nccl_GroupStart.group(1)
                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            ts_group_start[gpuId] = row[1] // 1000  ## ns to us
                            Parse_State[gpuId] = 1  ## awaiting ncclColl or ncclP2P, ignore ncclGroupStart/ncclGroupEnd in between

                        elif Parse_State[gpuId] == 2:
                            Parse_State[gpuId] = 3

                    elif match_nccl_GroupEnd:
                        pid = match_nccl_GroupEnd.group(1)
                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 3:
                            Parse_State[gpuId] = 2

                        elif Parse_State[gpuId] == 2:
                            ts_group_end[gpuId] = row[2] // 1000  ## ns to us
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['ts_end'] = ts_group_end[gpuId]
                            Parse_State[gpuId] = 4

                        elif Parse_State[gpuId] == 5:
                            ts_group_end[gpuId] = row[2] // 1000  ## ns to us
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['ts_end'] = ts_group_end[gpuId]
                            Parse_State[gpuId] = 6

                    elif match_nccl_Send:  ## 'ncclSend\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), receiver_rank (\d+), pid (\d+)'
                        comm = match_nccl_Send.group(1)
                        stream = match_nccl_Send.group(2)
                        data_size = int(match_nccl_Send.group(3))
                        type_size = int(match_nccl_Send.group(4))
                        peer_rank = match_nccl_Send.group(5)
                        pid = match_nccl_Send.group(6)

                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0
                        
                        if Parse_State[gpuId] == 1:
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1:
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

                        elif Parse_State[gpuId] == 2:
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]
                            streamId = stream_to_streamId[gpuId][stream]

                            if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId]['Send'] = {}

                            if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                            Parse_State[gpuId] = 2

                            last_P2P_streamId[gpuId] = streamId    
                            last_update[gpuId] = 'P2P'

                    elif match_nccl_Recv:
                        comm = match_nccl_Recv.group(1)
                        stream = match_nccl_Recv.group(2)
                        data_size = int(match_nccl_Recv.group(3))
                        type_size = int(match_nccl_Recv.group(4))
                        peer_rank = match_nccl_Recv.group(5)
                        pid = match_nccl_Recv.group(6)

                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6:
                            Parse_State[gpuId] = 0
                        
                        if Parse_State[gpuId] == 1:  ## 'ncclRecv\(\): comm (\S+), stream (\S+), data_size (\d+), type_size (\d+), sender_rank (\d+)'
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]

                            if comm_info[commId]['nranks'] > 1:
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

                        elif Parse_State[gpuId] == 2:
                            commId = comm_to_commId[gpuId][comm]
                            my_rank = comm_info[commId]['gpuId_To_rank'][gpuId]
                            streamId = stream_to_streamId[gpuId][stream]

                            if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                            if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                            Parse_State[gpuId] = 2

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

                        if Parse_State[gpuId] == 4:
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

                            events_counter[goal_rank][gpuId][commId][p2p_type][peer_rank] += 1

                    elif match_ncclLaunchKernel:
                        pid = match_ncclLaunchKernel.group(1)

                        gpuId = pid_to_gpuId[pid]

                        ts_kernel = row[2] // 1000 ## ns to us

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
                    gpuId = pid_to_gpuId[str(pid)]
                    if streamId not in cupti_kernel_results[goal_rank][gpuId]:
                        cupti_kernel_results[goal_rank][gpuId][streamId] = [] 

                    cupti_kernel_results[goal_rank][gpuId][streamId].append({
                        'gpu_event_type': fields[1],
                        'ts_gpu_start': start // 1000,
                        'ts_gpu_end': end // 1000,
                    })

            conn.close()
        
    return comm_init_events, nccl_events, cupti_kernel_results, comm_info, HostName_To_GoalRank

def events_list_equal(events_list_1, events_list_2):
    if len(events_list_1) != len(events_list_2):
        return 0
    
    else:
        num_events = len(events_list_1)
        for i in range(num_events):
            if events_list_1[i]['event_type'] != events_list_2[i]['gpu_event_type']:
                if not (events_list_1[i]['event_type'] == 'GroupColl' and events_list_1[i]['coll_type'] == events_list_2[i]['gpu_event_type']):
                    if not (events_list_1[i]['event_type'] == 'GroupP2P' and events_list_2[i]['gpu_event_type'] == 'SendRecv'):
                        return 0
                
        return 1        

def merge_nsys_events(nccl_events, cupti_kernel_results, comm_info):
    merged_events = {}
    for goal_rank, nccl_node_events in nccl_events.items():
        merged_events[goal_rank] = {}
        for gpuId, nccl_gpu_events in nccl_node_events.items():
            merged_events[goal_rank][gpuId] = {}
            for streamId, nccl_stream_events in nccl_gpu_events.items():
                merged_events[goal_rank][gpuId][streamId] = nccl_stream_events
                for gpu_streamId, cupti_stream_events in cupti_kernel_results[goal_rank][gpuId].items():
                    if events_list_equal(nccl_stream_events, cupti_stream_events):
                        for i in range(len(nccl_stream_events)):
                            merged_events[goal_rank][gpuId][streamId][i]['ts_gpu_start'] = cupti_stream_events[i]['ts_gpu_start']
                            merged_events[goal_rank][gpuId][streamId][i]['ts_gpu_end'] = cupti_stream_events[i]['ts_gpu_end']
                        
                        print(f'goal_rank: {goal_rank}, gpuId: {gpuId}, streamId: {streamId}, num_events: {len(merged_events[goal_rank][gpuId][streamId])}')

    return merged_events

def check_events_pair(events):
    events_pair = {}

    for goal_rank, goal_events in events.items():
        events_pair[goal_rank] = {}
        for gpuId, gpu_events in goal_events.items():
            events_pair[goal_rank][gpuId] = {}
            for streamId, stream_events in gpu_events.items():
                for event in stream_events:
                    if event['event_type'] not in events_pair[goal_rank][gpuId]:
                        events_pair[goal_rank][gpuId][event['event_type']] = {}

                    if event['commId'] not in events_pair[goal_rank][gpuId][event['event_type']]:
                        events_pair[goal_rank][gpuId][event['event_type']][event['commId']] = []

                    if streamId not in events_pair[goal_rank][gpuId][event['event_type']][event['commId']]:
                        events_pair[goal_rank][gpuId][event['event_type']][event['commId']].append(streamId)

    return events_pair

# def expand_group_events(events):
#     expanded_events = {}

#     for goal_rank, goal_events in events.items():
#         expanded_events[goal_rank] = {}
#         for gpuId, gpu_events in goal_events.items():
#             expanded_events[goal_rank][gpuId] = {}
#             for streamId, stream_events in gpu_events.items():
#                 expanded_events[goal_rank][gpuId][streamId] = []
#                 for event in stream_events:
#                     if event['event_type'] == 'GroupColl':
#                         for coll_event in event['coll_events']:
#                             expanded_events[goal_rank][gpuId][streamId].append(
#                                 {
#                                     'event_type': event['coll_type'],
#                                     'commId': event['commId'],
#                                     'comm_index': event['comm_index'],
#                                     'streamId': event['streamId'],
#                                     'my_rank': event['my_rank'],
#                                     'gpuId': event['gpuId'],
#                                     'ts_start': event['ts_start'],
#                                     'algorithm': coll_event['algorithm'],
#                                     'protocol': coll_event['protocol'],
#                                     'data_size': coll_event['data_size'],
#                                     'type_size': coll_event['type_size'],
#                                     'root': coll_event['root'],
#                                     'red_op': coll_event['red_op'],
#                                     'seq': coll_event['seq'],
#                                     'chunkSteps': coll_event['chunkSteps'],
#                                     'sliceSteps': coll_event['sliceSteps'],
#                                     'stepSize': coll_event['stepSize'],
#                                     'elems': coll_event['elems'],
#                                     'ts_end': event['ts_end'],
#                                     'ts_kernel': event['ts_kernel'],
#                                     'ts_gpu_start': event['ts_gpu_start'],
#                                     'ts_gpu_end': event['ts_gpu_end']
#                                 }
#                             )
                    
#                     elif event['event_type'] == 'GroupP2P':
#                         for p2p_event in event['P2P_events']:
#                             expanded_events[goal_rank][gpuId][streamId].append(
#                                 {
#                                     'event_type': p2p_event['p2p_type'],
#                                     'commId': event['commId'],
#                                     'comm_index': event['comm_index'],
#                                     'streamId': event['streamId'],
#                                     'my_rank': event['my_rank'],
#                                     'gpuId': event['gpuId'],
#                                     'ts_start': event['ts_start'],
#                                     'peer_rank': p2p_event['peer_rank'],
#                                     'protocol': p2p_event['protocol'],
#                                     'countHi32': p2p_event['countHi32'],
#                                     'countLo32': p2p_event['countLo32'],
#                                     'chunkSize': p2p_event['chunkSize'],
#                                     'count': p2p_event['count'],
#                                     'data_size': p2p_event['count'],
#                                     'seq': p2p_event['seq'],
#                                     'ts_end': event['ts_end'],
#                                     'ts_kernel': event['ts_kernel'],
#                                     'ts_gpu_start': event['ts_gpu_start'],
#                                     'ts_gpu_end': event['ts_gpu_end']
#                                 }
#                             )

#                     else:
#                         expanded_events[goal_rank][gpuId][streamId].append(event)

#     for goal_rank, goal_events in expanded_events.items():
#         for gpuId, gpu_events in goal_events.items():
#             for streamId, stream_events in gpu_events.items():
#                 print(f'goal_rank: {goal_rank}, gpuId: {gpuId}, streamId: {streamId}, num_events: {len(expanded_events[goal_rank][gpuId][streamId])}')

#     return expanded_events

def get_events_parallel_group(nccl_events):
    nccl_events_group = {}

    for goal_rank, goal_events in nccl_events.items():
        nccl_events_group[goal_rank] = {}
        for gpuId, gpu_events in goal_events.items():
            nccl_events_group[goal_rank][gpuId] = {}
            for streamId, stream_events in gpu_events.items():
                nccl_events_group[goal_rank][gpuId][streamId] = []
                for event_index, event in enumerate(stream_events):
                    if event['event_type'] == 'GroupColl':
                        events_group = {}    
                        events_group['events'] = []
                        events_group['ts_group_host_start'] = event['ts_start']
                        events_group['ts_group_gpu_end'] = event['ts_gpu_end']

                        for coll_event in event['coll_events']:
                            events_group['events'].append(
                                {
                                    'event_type': event['coll_type'],
                                    'commId': event['commId'],
                                    'comm_index': event['comm_index'],
                                    'streamId': event['streamId'],
                                    'my_rank': event['my_rank'],
                                    'gpuId': event['gpuId'],
                                    'ts_start': event['ts_start'],
                                    'algorithm': coll_event['algorithm'],
                                    'protocol': coll_event['protocol'],
                                    'data_size': coll_event['data_size'],
                                    'type_size': coll_event['type_size'],
                                    'root': coll_event['root'],
                                    'red_op': coll_event['red_op'],
                                    'seq': coll_event['seq'],
                                    'chunkSteps': coll_event['chunkSteps'],
                                    'sliceSteps': coll_event['sliceSteps'],
                                    'stepSize': coll_event['stepSize'],
                                    'elems': coll_event['elems'],
                                    'ts_end': event['ts_end'],
                                    'ts_kernel': event['ts_kernel'],
                                    'ts_gpu_start': event['ts_gpu_start'],
                                    'ts_gpu_end': event['ts_gpu_end']
                                }
                            )

                        nccl_events_group[goal_rank][gpuId][streamId].append(events_group)

                    elif event['event_type'] == 'GroupP2P':
                        events_group = {}    
                        events_group['events'] = []
                        events_group['ts_group_host_start'] = event['ts_start']
                        events_group['ts_group_gpu_end'] = event['ts_gpu_end']

                        for p2p_event in event['P2P_events']:
                            events_group['events'].append(
                                {
                                    'event_type': p2p_event['p2p_type'],
                                    'commId': event['commId'],
                                    'comm_index': event['comm_index'],
                                    'streamId': event['streamId'],
                                    'my_rank': event['my_rank'],
                                    'gpuId': event['gpuId'],
                                    'ts_start': event['ts_start'],
                                    'peer_rank': p2p_event['peer_rank'],
                                    'protocol': p2p_event['protocol'],
                                    'countHi32': p2p_event['countHi32'],
                                    'countLo32': p2p_event['countLo32'],
                                    'chunkSize': p2p_event['chunkSize'],
                                    'count': p2p_event['count'],
                                    'data_size': p2p_event['count'],
                                    'seq': p2p_event['seq'],
                                    'ts_end': event['ts_end'],
                                    'ts_kernel': event['ts_kernel'],
                                    'ts_gpu_start': event['ts_gpu_start'],
                                    'ts_gpu_end': event['ts_gpu_end']
                                }
                            )

                        nccl_events_group[goal_rank][gpuId][streamId].append(events_group)

                    else: 
                        events_group = {}    
                        events_group['events'] = []
                        events_group['ts_group_host_start'] = event['ts_start']
                        events_group['ts_group_gpu_end'] = event['ts_gpu_end']

                        events_group['events'].append(event)

                        nccl_events_group[goal_rank][gpuId][streamId].append(events_group)

    return nccl_events_group

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


def get_events_dependency(nccl_group_events, comm_init_events, goal_file_name):
    num_ranks = len(nccl_group_events)
    task_counter = 0
    with open(goal_file_name, 'w') as file:
        file.write(f"num_ranks {num_ranks}\n")

        for goal_rank in range(num_ranks):
            file.write(f"\nrank {goal_rank}")
            file.write(" {\n")

            goal_events = nccl_group_events[goal_rank]
            task_counter += 1
            file.write(f"l{task_counter}: calc 0\n") ## Start point of the node
            node_start_calc_id = task_counter
            
            task_counter += 1
            file.write(f"l{task_counter}: calc 0\n") ## End point of the node
            node_end_calc_id = task_counter

            for gpuId, gpu_events in goal_events.items():
                for streamId, stream_events in gpu_events.items():
                    last_group_event_end_time =  comm_init_events[goal_rank][gpuId]['ts_init_end']
                    last_group_event_end_id = node_start_calc_id
                    for group_event_index, group_event in enumerate(stream_events): 
                        task_counter += 1
                        file.write(f"l{task_counter}: calc {group_event['ts_group_host_start'] - last_group_event_end_time}\n")  ## Calc between first group host event start and last group gpu event end
                        file.write(f"l{task_counter} requires l{last_group_event_end_id}\n")
                        group_event_start_calc_id = task_counter

                        task_counter += 1
                        file.write(f"l{task_counter}: calc 0\n")  ## End calc of the parallel group of events
                        group_event_end_calc_id = task_counter
                        last_group_event_end_time = group_event['ts_group_gpu_end']
                        last_group_event_end_id = task_counter

                        for event in group_event['events']:
                            if event['event_type'] == 'GroupP2P':
                                task_counter += 1
                                file.write(f"l{task_counter}: calc {event['ts_kernel'] - event['ts_start']}\n")  ## Calc between nccl kernel launch end and host event start
                                file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                p2p_group_start_calc_id = task_counter

                                task_counter += 1
                                file.write(f"l{task_counter}: calc 0\n")
                                file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")
                                p2p_group_end_calc_id = task_counter

                                for p2p_event in event['P2P_events']:
                                    task_counter += 1
                                    file.write(f"l{task_counter}: {p2p_event['event_type']} {p2p_event['data_size']} bytes peer {p2p_event['peer_rank']} comm {event['comm_index']} gpu {gpuId} stream {streamId} seq {p2p_event['seq']} end\n")
                                    file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                                    file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")
                            
                            else:
                                task_counter += 1
                                file.write(f"l{task_counter}: calc {event['ts_kernel'] - event['ts_start']}\n")  ## Calc between nccl kernel launch end and host event start
                                file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")

                                task_counter += 1
                                file.write(f"l{task_counter}: {event['event_type']} {event['data_size']} bytes comm {event['comm_index']} gpu {gpuId} stream {streamId} seq {event['seq']} end\n")  ## gpu event
                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")                     

                        if group_event_index == len(stream_events) - 1:
                            file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")

            file.write("}\n")

def modRanks(r, nranks):
    return r - nranks if r >= nranks else r     

def div_up(x, y):
    return (x + y - 1) // y       

# def get_reduction_time(data_size):
#     return data_size//10  ## us

def get_reduction_time(data_size, protocol):  ## data_size: size of data, not data + flag
    if protocol == '2':
        with importlib.resources.open_text("nccl_goal_generator", "npkit_data_summary_Simple.json") as file:
            data = json.load(file)

    elif protocol == '0':
        with importlib.resources.open_text("nccl_goal_generator", "npkit_data_summary_LL.json") as file:
            data = json.load(file)

    if str(data_size) in data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND']:
        reduction_times = data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(data_size)]
        return random.choice(reduction_times)

    sizes = [int(size) for size in data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'].keys()]
    sizes.sort()

    if data_size < sizes[0]:
        return random.choice(data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(sizes[0])])
    if data_size > sizes[-1]:
        return random.choice(data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(sizes[-1])])

    f = interpolate.interp1d(sizes, [np.mean(data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(size)]) for size in sizes], kind='linear', fill_value="extrapolate")
    interpolated_value = f(data_size)
    
    return int(random.gauss(interpolated_value, interpolated_value * 0.1))

# def get_copy_time(data_size):
#     return data_size//10  ## us

def get_copy_time(data_size, protocol):  ## data_size: size of data, not data + flag
    if protocol == '2':
        with importlib.resources.open_text("nccl_goal_generator", "npkit_data_summary_Simple.json") as file:
            data = json.load(file)

    elif protocol == '0':
        with importlib.resources.open_text("nccl_goal_generator", "npkit_data_summary_LL.json") as file:
            data = json.load(file)

    if str(data_size) in data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND']:
        reduction_times = data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(data_size)]
        return random.choice(reduction_times)

    sizes = [int(size) for size in data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'].keys()]
    sizes.sort()

    if data_size < sizes[0]:
        return random.choice(data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(sizes[0])])
    if data_size > sizes[-1]:
        return random.choice(data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(sizes[-1])])

    f = interpolate.interp1d(sizes, [np.mean(data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(size)]) for size in sizes], kind='linear', fill_value="extrapolate")
    interpolated_value = f(data_size)
    
    return int(random.gauss(interpolated_value, interpolated_value * 0.1))

def get_intra_node_gpu_transfer_time(data_size, transfer_type):  
    """
    data_size [byte]: data + flag, not just data
    Here we use the bandwidth of NVLink 1.0, which is 20 GT/s , each transfer is 1 byte
    """

    if transfer_type == 'Send':
        return data_size * 10**6 // (20 * 10**9 * 1)  ## us

    elif transfer_type == 'Recv':
        return 0

def get_event_type(operation):
    if operation == 'AllReduce':
        return 0
    elif operation == 'Broadcast':
        return 1
    elif operation == 'AllGather':
        return 2
    if operation == 'ReduceScatter':
        return 3
    elif operation == 'Send':
        return 5
    elif operation == 'Recv':
        return 5

def get_in_gpu_microevents_dependency(nccl_group_events, comm_init_events, comm_info, goal_file_name):
    num_ranks = len(nccl_group_events)
    task_counter = 0
    SendRecvEvents_To_TaskCounter = {}
    with open(goal_file_name, 'w') as file:
        file.write(f"num_ranks {num_ranks}\n")

        for goal_rank in range(num_ranks):
            file.write(f"\nrank {goal_rank}")
            file.write(" {\n")

            SendRecvEvents_To_TaskCounter[goal_rank] = {}

            goal_events = nccl_group_events[goal_rank]
            task_counter += 1
            file.write(f"l{task_counter}: calc 0\n") ## Start point of the node
            node_start_calc_id = task_counter
            
            task_counter += 1
            file.write(f"l{task_counter}: calc 0\n") ## End point of the node
            node_end_calc_id = task_counter

            for gpuId, gpu_events in goal_events.items():
                SendRecvEvents_To_TaskCounter[goal_rank][gpuId] = {}
                for streamId, stream_events in gpu_events.items():
                    last_group_event_end_time =  comm_init_events[goal_rank][gpuId]['ts_init_end']
                    last_group_event_end_id = node_start_calc_id
                    for group_event_index, group_event in enumerate(stream_events): 
                        launched = 0

                        task_counter += 1
                        file.write(f"l{task_counter}: calc {max(group_event['ts_group_host_start'] - last_group_event_end_time, 0)}\n")  ## Calc between first group host event start and last group gpu event end
                        file.write(f"l{task_counter} requires l{last_group_event_end_id}\n")
                        group_event_start_calc_id = task_counter

                        task_counter += 1
                        file.write(f"l{task_counter}: calc 0\n")  ## End calc of the parallel group of events
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
                                    file.write(f"l{task_counter}: calc {event['ts_kernel'] - event['ts_start']}\n")  ## Calc between nccl kernel launch end and host event start
                                    file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                    p2p_group_start_calc_id = task_counter

                                    task_counter += 1
                                    file.write(f"l{task_counter}: calc 0\n")
                                    file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")
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
                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][p2p_event_type][p2p_peer_Ix][p2p_seq][channel_id])) + str(channel_id).zfill(2) + str(p2p_seq).zfill(4) + str(get_event_type(p2p_event_type)).zfill(1) + str(event['comm_index']).zfill(2)
                                        if p2p_event_type == 'Send':
                                            file.write(f"l{task_counter}: send {nelem}b to {p2p_peer_Ix} tag {tag}\n")
                                        elif p2p_event_type == 'Recv':
                                            file.write(f"l{task_counter}: recv {nelem}b from {p2p_peer_Ix} tag {tag}\n")
                                        file.write(f"l{task_counter} requires l{p2p_group_start_calc_id}\n")
                                        file.write(f"l{p2p_group_end_calc_id} requires l{task_counter}\n")

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
                                    file.write(f"l{task_counter}: calc {event['ts_kernel'] - event['ts_start']}\n")  ## Calc between nccl kernel launch end and host event start
                                    file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                    gpu_event_start_calc_id = task_counter

                                    task_counter += 1
                                    file.write(f"l{task_counter}: calc 0\n")  ## end calc of a gpu event
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
                                                    file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                
                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                    file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto)}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto)}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                
                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                    file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)}\n")
                                                        calc_task_id = task_counter

                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {child_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)

                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {child_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)}\n")
                                                                calc_task_id = task_counter

                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        task_counter += 1
                                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                        file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {child_Ix} tag {tag}\n")
                                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                        file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)

                                                                        task_counter += 1
                                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                        file.write(f"l{task_counter}: send {sliceSize * type_size}b to {child_Ix} tag {tag}\n")
                                                                        file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {parent_Ix} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                        task_counter += 1  ## Recv
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {parent_Ix} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {parent_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                                task_counter += 1  ## Recv
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {parent_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                                        calc_task_id = task_counter

                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {child_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)
                                    
                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {parent_Ix} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                        ## RecvCopySend
                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto)}\n")
                                                        calc_task_id = task_counter

                                                        task_counter += 1
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {parent_Ix} tag {tag}\n")
                                                        file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix].append(task_counter)
                                                        
                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {child_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                                calc_task_id = task_counter

                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        task_counter += 1
                                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                        file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {child_Ix} tag {tag}\n")
                                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                        file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][child_Ix].append(task_counter)
                                            
                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {parent_Ix} tag {tag}\n")
                                                                file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][parent_Ix].append(task_counter)

                                                                ## RecvCopySend
                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto)}\n")
                                                                calc_task_id = task_counter

                                                                task_counter += 1
                                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {parent_Ix} tag {tag}\n")
                                                                file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][parent_Ix].append(task_counter)
                                                                
                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        task_counter += 1
                                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][child_Ix])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                        file.write(f"l{task_counter}: send {sliceSize * type_size}b to {child_Ix} tag {tag}\n")
                                                                        file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                        loopCount = nranks * chunkCount

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            # offset = gridOffset + elemOffset
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            if (ringIx == root_rank):  ## Send
                                                if proto == '0':
                                                    # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            elif nextIx == root_rank: ## Recv
                                                if proto == '0':
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break
                                                
                                            else:  ## RecvCopySend
                                                if proto == '0':
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            ## step 0: Send
                                            if proto == '0':
                                                # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                    file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {sliceSize}b to {nextIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            ## Step (k - 1): Recv
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {prevIx} tag {tag}\n")
                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {sliceSize}b from {prevIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                        file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                    file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {nextIx} tag {tag}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {sliceSize * type_size}b to {nextIx} tag {tag}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['send'][nextIx].append(task_counter)
                                                            
                                                            slice += 1
                                                            offset += sliceSize

                                                            if not (slice < SlicePerChunk and offset < nelem):
                                                                break

                                            ## Step (k - 1): RecvReduceCopy
                                            if proto == '0':
                                                task_counter += 1
                                                tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {prevIx} tag {tag}\n")
                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                task_counter += 1
                                                file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)}\n")
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
                                                        tag = str(len(SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx])) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {prevIx} tag {tag}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        SendRecvEvents_To_TaskCounter[goal_rank][gpuId][commId][event['event_type']][event['seq']][channel_id]['recv'][prevIx].append(task_counter)

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)}\n")
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

                        if group_event_index == len(stream_events) - 1:
                            file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")

            file.write("}\n")

    return SendRecvEvents_To_TaskCounter

def get_inter_node_microevents_dependency(nccl_group_events, comm_init_events, comm_info, SendRecvEvents_To_TaskCounter, goal_file_name):
    num_ranks = len(nccl_group_events)
    task_counter = 0
    with open(goal_file_name, 'w') as file:
        file.write(f"num_ranks {num_ranks}\n")

        for goal_rank in range(num_ranks):
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

            for gpuId, gpu_events in goal_events.items():
                gpuId_commId_cpu_counter[gpuId] = {}

                for streamId, stream_events in gpu_events.items():
                    cpu_counter_start = cpu_counter_end + 1
                    cpu_counter = cpu_counter_start
                    cpu_counter_end = cpu_counter

                    last_group_event_end_time =  comm_init_events[goal_rank][gpuId]['ts_init_end']
                    last_group_event_end_id = node_start_calc_id
                    for group_event_index, group_event in enumerate(stream_events): 
                        launched = 0
                        cpu_counter = cpu_counter_start

                        task_counter += 1
                        file.write(f"l{task_counter}: calc {max(group_event['ts_group_host_start'] - last_group_event_end_time, 0)} cpu {cpu_counter}\n")  ## Calc between first group host event start and last group gpu event end
                        file.write(f"l{task_counter} requires l{last_group_event_end_id}\n")
                        group_event_start_calc_id = task_counter

                        task_counter += 1
                        file.write(f"l{task_counter}: calc 0 cpu {cpu_counter}\n")  ## End calc of the parallel group of events
                        group_event_end_calc_id = task_counter
                        last_group_event_end_time = group_event['ts_group_gpu_end']
                        last_group_event_end_id = task_counter

                        for event_index, event in enumerate(group_event['events']):
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
                                    file.write(f"l{task_counter}: calc {event['ts_kernel'] - event['ts_start']} cpu {cpu_counter}\n")  ## Calc between nccl kernel launch end and host event start
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
                                                file.write(f"l{task_counter}: send {nelem}b to {goal_rank_peer} tag {tag} cpu {cpu_counter}\n")
                                                p2p_index[p2p_peer_Ix] += 1
                                            else:
                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(nelem, 'Send')} cpu {cpu_counter}\n")

                                        elif p2p_event_type == 'Recv':
                                            if goal_rank_peer != goal_rank:
                                                tag = str(p2p_index[p2p_peer_Ix]) + str(channel_id).zfill(2) + str(p2p_seq).zfill(4) + str(get_event_type(p2p_event_type)).zfill(1) + str(event['comm_index']).zfill(2)
                                                file.write(f"l{task_counter}: recv {nelem}b from {goal_rank_peer} tag {tag} cpu {cpu_counter}\n")
                                                p2p_index[p2p_peer_Ix] += 1
                                            else:
                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(nelem, 'Recv')} cpu {cpu_counter}\n")

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
                                    file.write(f"l{task_counter}: calc {event['ts_kernel'] - event['ts_start']} cpu {cpu_counter}\n")  ## Calc between nccl kernel launch end and host event start
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
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[prevIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                    file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    recv_index[prevIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                
                                                                task_counter += 1
                                                                if goal_rank_next != goal_rank:
                                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1
                                                    
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[prevIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        
                                                        task_counter += 1
                                                        if goal_rank_next != goal_rank:
                                                            tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                    file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    recv_index[prevIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                
                                                                task_counter += 1
                                                                if goal_rank_next != goal_rank:
                                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[nextIx] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                recv_index[prevIx] += 1
                                                            
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
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
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                        calc_task_id = task_counter

                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']
                                                                
                                                                task_counter += 1
                                                                if goal_rank != goal_rank_child:
                                                                    tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    recv_index[child_Ix] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")

                                                                task_counter += 1
                                                                if goal_rank != goal_rank_child:
                                                                    tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[child_Ix] += 1

                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                                calc_task_id = task_counter

                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                        goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']
                                                                        
                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                            recv_index[child_Ix] += 1

                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")

                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                            send_index[child_Ix] += 1

                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[parent_Ix] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        task_counter += 1  ## Recv
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            recv_index[parent_Ix] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
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
                                                                    file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                                task_counter += 1  ## Recv
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    recv_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
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
                                                        file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                        calc_task_id = task_counter

                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                task_counter += 1
                                                                if goal_rank_child != goal_rank:
                                                                    tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    recv_index[child_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                    
                                                        task_counter += 1
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(send_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[parent_Ix] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        ## RecvCopySend
                                                        task_counter += 1
                                                        file.write(f"l{task_counter}: calc {get_copy_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                        calc_task_id = task_counter

                                                        task_counter += 1
                                                        if goal_rank_parent != goal_rank:
                                                            tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                            file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[parent_Ix] += 1

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        
                                                        for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                            if child_Ix != '-1':
                                                                gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                task_counter += 1
                                                                if goal_rank_child != goal_rank:
                                                                    tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[child_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                                calc_task_id = task_counter

                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                        goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(recv_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                            recv_index[child_Ix] += 1
                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                            file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                            
                                                                task_counter += 1
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(send_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                    send_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                                ## RecvCopySend
                                                                task_counter += 1
                                                                file.write(f"l{task_counter}: calc {get_copy_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                                calc_task_id = task_counter

                                                                task_counter += 1
                                                                if goal_rank_parent != goal_rank:
                                                                    tag = str(recv_index[parent_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                    file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_parent} tag {tag} cpu {cpu_counter}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                    recv_index[parent_Ix] += 1
                                                                else:
                                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                    file.write(f"l{calc_task_id} requires l{task_counter}\n")
                                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                
                                                                for child_Ix in [child_1_Ix, child_2_Ix, child_3_Ix]:
                                                                    if child_Ix != '-1':
                                                                        gpuId_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['gpuId']
                                                                        goal_rank_child = comm_info[commId]['rank_To_rankInfo'][child_Ix]['goal_rank']

                                                                        task_counter += 1
                                                                        if goal_rank_child != goal_rank:
                                                                            tag = str(send_index[child_Ix]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                            file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_child} tag {tag} cpu {cpu_counter}\n")
                                                                            file.write(f"l{task_counter} requires l{calc_task_id}\n")
                                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                            send_index[child_Ix] += 1
                                                                        else:
                                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                        loopCount = nranks * chunkCount

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            # offset = gridOffset + elemOffset
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            if (ringIx == root_rank):  ## Send
                                                if proto == '0':
                                                    # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: send {sliceSize}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send')} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                recv_index[prevIx] += 1
                                                            
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv')} cpu {cpu_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1

                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv')} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1

                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send')} cpu {cpu_counter}\n")
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

                                        for elemOffset in range(0, channelCount, chunkCount):
                                            nelem = int(min(chunkCount, channelCount - elemOffset))
                                            nelem = 0 if nelem < 0 else nelem

                                            ## step 0: Send
                                            if proto == '0':
                                                # EltPerLine = 8 // type_size ## sizeof(uint64_t)： 8 bytes
                                                task_counter += 1
                                                if goal_rank_next != goal_rank:
                                                    tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                    file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    send_index[nextIx] += 1
                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: send {sliceSize}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send')} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_copy_time(nelem, proto)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv')} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_copy_time(sliceSize, proto)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Send')} cpu {cpu_counter}\n")
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
                                                    file.write(f"l{task_counter}: recv {div_up(nelem, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    recv_index[prevIx] += 1
                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            recv_index[prevIx] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize, 'Recv')} cpu {cpu_counter}\n")
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
                                                    file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                    send_index[nextIx] += 1
                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                            send_index[nextIx] += 1
                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                        file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                        recv_index[prevIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    
                                                    task_counter += 1
                                                    if goal_rank_next != goal_rank:
                                                        tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                        file.write(f"l{task_counter}: send {div_up(nelem * type_size, 8) * 16}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                        file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                        file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                        send_index[nextIx] += 1
                                                    else:
                                                        file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Send')} cpu {cpu_counter}\n")
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
                                                                file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                                recv_index[prevIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            
                                                            task_counter += 1
                                                            if goal_rank_next != goal_rank:
                                                                tag = str(send_index[nextIx]) + str(channel_id).zfill(2) + str(event['seq']).zfill(4) + str(get_event_type(event['event_type'])).zfill(1) + str(event['comm_index']).zfill(2)
                                                                file.write(f"l{task_counter}: send {sliceSize * type_size}b to {goal_rank_next} tag {tag} cpu {cpu_counter}\n")
                                                                file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                                file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")
                                                                send_index[nextIx] += 1
                                                            else:
                                                                file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Send')} cpu {cpu_counter}\n")
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
                                                    file.write(f"l{task_counter}: recv {div_up(nelem * type_size, 8) * 16}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                    recv_index[prevIx] += 1

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                    file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                else:
                                                    file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(div_up(nelem * type_size, 8) * 16, 'Recv')} cpu {cpu_counter}\n")
                                                    file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                    task_counter += 1
                                                    file.write(f"l{task_counter}: calc {get_reduction_time(nelem * type_size, proto) + get_copy_time(nelem * type_size, proto)} cpu {cpu_counter}\n")
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
                                                            file.write(f"l{task_counter}: recv {sliceSize * type_size}b from {goal_rank_prev} tag {tag} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")
                                                            recv_index[prevIx] += 1

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{task_counter - 1}\n")
                                                            file.write(f"l{gpu_event_end_calc_id} requires l{task_counter}\n")

                                                        else:
                                                            file.write(f"l{task_counter}: calc {get_intra_node_gpu_transfer_time(sliceSize * type_size, 'Recv')} cpu {cpu_counter}\n")
                                                            file.write(f"l{task_counter} requires l{gpu_event_start_calc_id}\n")

                                                            task_counter += 1
                                                            file.write(f"l{task_counter}: calc {get_reduction_time(sliceSize * type_size, proto) + get_copy_time(sliceSize * type_size, proto)} cpu {cpu_counter}\n")
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

                        if group_event_index == len(stream_events) - 1:
                            file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")
            
            for gpuId, gpu_events in goal_events.items():
                for streamId, stream_events in gpu_events.items():
                    for group_event_index, group_event in enumerate(stream_events): 
                        for event in group_event['events']:
                            if event['event_type'] == 'AllReduce' or event['event_type'] == 'Broadcast' or event['event_type'] == 'AllGather' or event['event_type'] == 'ReduceScatter':
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

def main(config_node_gpu=None, results_dir=None):
    # Get nsys events
    Dir_Path = results_dir
    nsys_reports_path = os.path.join(Dir_Path, 'nsys_reports')
    Comm_Init_Events, NCCL_Events, CUPTI_Kernel_Results, Comm_Info, HostName_To_GoalRank = get_nsys_events(nsys_reports_path)  ## nccl_events, cupti_kernel_results, comm_info, HostName_To_GoalRank
    file_path = os.path.join(Dir_Path, 'Events_Dependency.goal')
    with open(file_path, 'w') as json_file:
        json.dump(HostName_To_GoalRank, json_file, indent=4)
        json_file.write("\n\n")
        json.dump(Comm_Info, json_file, indent=4)
        json_file.write("\n\n")
        json.dump(CUPTI_Kernel_Results, json_file, indent=4)
        json_file.write("\n\n")
        json.dump(NCCL_Events, json_file, indent=4)
        json_file.write("\n\n")
        json.dump(Comm_Init_Events, json_file, indent=4)
    print('Nsys_Events has been exported to nsys_events_intermediate_output.json')

    Merged_Events = merge_nsys_events(NCCL_Events, CUPTI_Kernel_Results, Comm_Info)
    file_path = os.path.join(Dir_Path, 'nsys_events_merged_output.json')
    with open(file_path, 'w') as json_file:
        json.dump(Merged_Events, json_file, indent=4)
        json_file.write("\n\n")
    print('Merged_Events has been exported to nsys_events_merged_output.json')

    Events_Pair = check_events_pair(Merged_Events)
    file_path = os.path.join(Dir_Path, 'nsys_events_pair_output.json')
    with open(file_path, 'w') as json_file:
        json.dump(Events_Pair, json_file, indent=4)
        json_file.write("\n\n")

    # Expanded_Events = expand_group_events(Merged_Events)
    # file_path = os.path.join(Dir_Path, 'nsys_events_expanded_output.json')
    # with open(file_path, 'w') as json_file:
    #     json.dump(Expanded_Events, json_file, indent=4)
    #     json_file.write("\n\n")

    Events_Parallel_Group = get_events_parallel_group(Merged_Events)
    file_path = os.path.join(Dir_Path, 'nsys_events_parallel_group_output.json')
    with open(file_path, 'w') as json_file:
        json.dump(Events_Parallel_Group, json_file, indent=4)
        json_file.write("\n\n")

    if config_node_gpu is not None:
        Events_Parallel_Group, Comm_Init_Events, Comm_Info = apply_user_config(config_node_gpu, Events_Parallel_Group, Comm_Init_Events, Comm_Info)

    Goal_File_Name = os.path.join(Dir_Path, 'Events_Dependency.goal')
    get_events_dependency(Events_Parallel_Group, Comm_Init_Events, Goal_File_Name)
    print('Events goal file has been exported to Events_Dependency.goal')

    Goal_File_Name = os.path.join(Dir_Path, 'InGPU_MicroEvents_Dependency.goal')
    SendRecvEvents_To_TaskCounter = get_in_gpu_microevents_dependency(Events_Parallel_Group, Comm_Init_Events, Comm_Info, Goal_File_Name)
    file_path = os.path.join(Dir_Path, 'SendRecvEvents_To_TaskCounter.json')
    with open(file_path, 'w') as json_file:
        json.dump(SendRecvEvents_To_TaskCounter, json_file, indent=4)
        json_file.write("\n\n")
    print('In-GPU goal file has been exported to InGPU_MicroEvents_Dependency.goal')

    Goal_File_Name = os.path.join(Dir_Path, 'InterNode_MicroEvents_Dependency.goal')
    get_inter_node_microevents_dependency(Events_Parallel_Group, Comm_Init_Events, Comm_Info, SendRecvEvents_To_TaskCounter, Goal_File_Name)
    print('Internode goal file has been exported to InterNode_MicroEvents_Dependency.goal')

if __name__ == '__main__':
    main()