def events_list_equal(events_list_1, events_list_2):
    if len(events_list_1) != len(events_list_2):
        return 0
    
    
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
                matched = False
                for gpu_streamId, cupti_stream_events in cupti_kernel_results[goal_rank][gpuId].items():
                    if events_list_equal(nccl_stream_events, cupti_stream_events):
                        matched = True
                        for i in range(len(nccl_stream_events)):
                            merged_events[goal_rank][gpuId][streamId][i]['ts_gpu_start'] = cupti_stream_events[i]['ts_gpu_start']
                            merged_events[goal_rank][gpuId][streamId][i]['ts_gpu_end'] = cupti_stream_events[i]['ts_gpu_end']
                        
                        print(f'goal_rank: {goal_rank}, gpuId: {gpuId}, streamId: {streamId}, gpu_streamId: {gpu_streamId}, num_events: {len(merged_events[goal_rank][gpuId][streamId])}')
                if not matched:
                    print(f"[ERROR] goal_rank: {goal_rank}, gpuId: {gpuId}, streamId: {streamId} has no matching cupti events")

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
                        events_group['ts_group_gpu_start'] = event['ts_gpu_start']
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
                                    'root_rank': coll_event['root_rank'],
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
                        events_group['ts_group_gpu_start'] = event['ts_gpu_start']
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
                        events_group['ts_group_gpu_start'] = event['ts_gpu_start']
                        events_group['ts_group_gpu_end'] = event['ts_gpu_end']

                        events_group['events'].append(event)

                        nccl_events_group[goal_rank][gpuId][streamId].append(events_group)

    return nccl_events_group
