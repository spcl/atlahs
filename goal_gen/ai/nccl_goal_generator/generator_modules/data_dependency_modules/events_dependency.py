from .utils import modRanks, div_up, get_event_type
from .intra_node_gpu_transfer_time import get_intra_node_gpu_transfer_time
from .reduction_copy_time import get_reduction_time, get_copy_time

def get_events_dependency(nccl_group_events, comm_init_events,
                          goal_file_name, profile_interval=None):
    num_ranks = len(nccl_group_events)
    # task_counter = 0
    with open(goal_file_name, 'w') as file:
        file.write(f"num_ranks {num_ranks}\n")

        for goal_rank in range(num_ranks):
            task_counter = 0
            
            file.write(f"\nrank {goal_rank}")
            file.write(" {\n")

            goal_events = nccl_group_events[goal_rank]
            print(f"[DEBUG] Goal Rank: {goal_rank}")
            # print(goal_events)
            task_counter += 1
            file.write(f"l{task_counter}: calc 0\n") ## Start point of the node
            node_start_calc_id = task_counter
            
            task_counter += 1
            file.write(f"l{task_counter}: calc 0\n") ## End point of the node
            node_end_calc_id = task_counter
            profile_start = 0
            profile_end = float('inf')

            for gpuId, gpu_events in goal_events.items():
                if gpuId in profile_interval:
                    profile_start = profile_interval[gpuId]["start"]
                    profile_end = profile_interval[gpuId]["end"]
                    print(f"[DEBUG] Profile Interval: {profile_interval[gpuId]}")
                
                gpu_all_stream_start_time = None
                for streamId, stream_events in gpu_events.items():
                    if gpu_all_stream_start_time is None:
                        gpu_all_stream_start_time = stream_events[0]['ts_group_gpu_start']
                    else:
                        gpu_all_stream_start_time = min(gpu_all_stream_start_time, stream_events[0]['ts_group_gpu_start'])
                        
                for streamId, stream_events in gpu_events.items():
                    last_group_event_end_time =  comm_init_events[goal_rank][gpuId]['ts_init_end']
                    last_group_event_end_id = node_start_calc_id
                    for group_event_index, group_event in enumerate(stream_events): 
                        launched = 0

                        task_counter += 1
                        file.write(f"l{task_counter}: calc {group_event['ts_group_gpu_start'] - last_group_event_end_time}\n")  ## Former calc between first group host event start and last group gpu event end
                        file.write(f"l{task_counter} requires l{last_group_event_end_id}\n")
                        group_event_start_calc_id = task_counter

                        task_counter += 1
                        file.write(f"l{task_counter}: calc 0\n")  ## End calc of the parallel group of events
                        group_event_end_calc_id = task_counter
                        last_group_event_end_time = group_event['ts_group_gpu_end']
                        last_group_event_end_id = task_counter

                        for event in group_event['events']:
                            if launched == 0:
                                task_counter += 1
                                file.write(f"l{task_counter}: calc 0\n")  ## Former calc between nccl kernel launch end and host event start
                                file.write(f"l{task_counter} requires l{group_event_start_calc_id}\n")
                                group_start_calc_id = task_counter

                                task_counter += 1
                                file.write(f"l{task_counter}: calc 0\n")
                                file.write(f"l{group_event_end_calc_id} requires l{task_counter}\n")
                                group_end_calc_id = task_counter

                                launched = 1

                            task_counter += 1
                            file.write(f"l{task_counter}: {event['event_type']} {event['data_size']} bytes comm {event['comm_index']} gpu {gpuId} stream {streamId} seq {event['seq']} end\n")  ## gpu event
                            file.write(f"l{task_counter} requires l{group_start_calc_id}\n")
                            file.write(f"l{group_end_calc_id} requires l{task_counter}\n")                     

                        if group_event_index == len(stream_events) - 1:
                            file.write(f"l{node_end_calc_id} requires l{last_group_event_end_id}\n")

            file.write("}\n")
