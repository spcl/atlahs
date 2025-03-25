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
from collections import defaultdict
from queue import Queue
from tqdm import tqdm

from generator_modules.nsys_events import get_nsys_events, merge_stream_if_no_overlap
from generator_modules.manipulate_events import merge_nsys_events, check_events_pair, get_events_parallel_group

from generator_modules.apply_config import apply_user_config
from generator_modules.data_dependency_modules.events_dependency import get_events_dependency
from generator_modules.data_dependency_modules.in_gpu_dependency import get_in_gpu_microevents_dependency
from generator_modules.data_dependency_modules.inter_node_dependency import get_inter_node_microevents_dependency
from generator_modules.data_dependency_modules.reduction_copy_time import init_data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--trace-dir", type=str, required=True,
                        help="Directory containing the nsys profiles of the application")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory to store the goal files and intermediate files")
    parser.add_argument("-q", "--no-intermediate-output", action="store_true",
                        help="Do not generate intermediate output files")

    parser.add_argument(
        '-c', '--config_node_gpu',
        type=str, 
        required=False, 
        help='yaml file for configuration of nodes and GPUs'
    )

    parser.add_argument(
        '-s', '--npkit_file_Simple',
        type=str, 
        default="npkit_benchmark_results/clariden/npkit_data_summary_Simple.json",
        help='NPKit benchmark results json file for Simple Protocol'
    )

    parser.add_argument(
        '-l', '--npkit_file_LL',
        type=str, 
        default="npkit_benchmark_results/clariden/npkit_data_summary_LL.json",
        help='NPKit benchmark results json file for LL Protocol'
    )

    parser.add_argument('--zero-red-copy', action='store_true', help='Whether to set all the reduction copy time to zero')
    parser.add_argument('--merge-non-overlap', action='store_true', help='Whether to merge non-overlapping events for all streams if possible')
    parser.add_argument("--unique-nic", action='store_true', help="Whether to assign a separate NIC ID for each GPU in GOAL")
    args = parser.parse_args()

    if args.no_intermediate_output:
        print("[INFO] No intermediate output will be generated")
    init_data(args.npkit_file_Simple, args.npkit_file_LL)

    # Get nsys events
    Dir_Path = args.trace_dir
    assert os.path.exists(Dir_Path), f"Directory {Dir_Path} does not exist"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Comm_Init_Events, NCCL_Events, CUPTI_Kernel_Results, Comm_Info, \
        HostName_To_GoalRank, profile_interval = get_nsys_events(Dir_Path)  ## nccl_events, cupti_kernel_results, comm_info, HostName_To_GoalRank
    
    if args.merge_non_overlap:
        NCCL_Events, CUPTI_Kernel_Results = merge_stream_if_no_overlap(NCCL_Events, CUPTI_Kernel_Results)
    
    intermediate_output = {
        "hostname_to_rank": HostName_To_GoalRank,
        "comm_info": Comm_Info,
        "cupti_kernel_results": CUPTI_Kernel_Results,
        "nccl_events": NCCL_Events,
        "comm_init_events": Comm_Init_Events,
        "profile_interval": profile_interval
    }

    if not args.no_intermediate_output:
        out_json_path = os.path.join(output_dir, 'nsys_events_intermediate_output.json')
        with open(out_json_path, 'w') as json_file:
            json.dump(intermediate_output, json_file, indent=4)
        print('Nsys_Events has been exported to nsys_events_intermediate_output.json')
    # exit(0)
    Merged_Events = merge_nsys_events(NCCL_Events, CUPTI_Kernel_Results, Comm_Info)

    if not args.no_intermediate_output:
        out_json_path = os.path.join(output_dir, 'nsys_events_merged_output.json')
        with open(out_json_path, 'w') as json_file:
            json.dump(Merged_Events, json_file, indent=4)
            json_file.write("\n\n")
        print('Merged_Events has been exported to nsys_events_merged_output.json')

    Events_Pair = check_events_pair(Merged_Events)
    
    if not args.no_intermediate_output:
        out_json_path = os.path.join(output_dir, 'nsys_events_pair_output.json')
        with open(out_json_path, 'w') as json_file:
            json.dump(Events_Pair, json_file, indent=4)
            json_file.write("\n\n")

    # Expanded_Events = expand_group_events(Merged_Events)
    # with open('./results/nsys_events_expanded_output.json', 'w') as json_file:
    #     json.dump(Expanded_Events, json_file, indent=4)
    #     json_file.write("\n\n")

    Events_Parallel_Group = get_events_parallel_group(Merged_Events)
    if not args.no_intermediate_output:
        out_json_path = os.path.join(output_dir, 'nsys_events_parallel_group_output.json')
        with open(out_json_path, 'w') as json_file:
            json.dump(Events_Parallel_Group, json_file, indent=4)
            json_file.write("\n\n")

    if args.config_node_gpu is not None:
        Events_Parallel_Group, Comm_Init_Events, Comm_Info = apply_user_config(args.config_node_gpu, Events_Parallel_Group, Comm_Init_Events, Comm_Info)
        if not args.no_intermediate_output:
            out_json_path = os.path.join(output_dir, 'nsys_events_user_config_output.json')
            with open(out_json_path, 'w') as json_file:
                json.dump(Comm_Info, json_file, indent=4)
                json_file.write("\n\n")

    # Goal_File_Name = './results/Events_Dependency.goal'
    if not args.no_intermediate_output:
        Goal_File_Name = os.path.join(output_dir, 'Events_Dependency.goal')
        get_events_dependency(Events_Parallel_Group, Comm_Init_Events, Goal_File_Name, profile_interval)
        print('Events goal file has been exported to Events_Dependency.goal')

    print(f"[INFO] Start to generate goal file for In-GPU and Internode events")
    # Goal_File_Name = './results/InGPU_MicroEvents_Dependency.goal'
    Goal_File_Name = os.path.join(output_dir, 'InGPU_MicroEvents_Dependency.goal')
    SendRecvEvents_To_TaskCounter = get_in_gpu_microevents_dependency(Events_Parallel_Group, Comm_Init_Events, Comm_Info, Goal_File_Name, profile_interval, True)

    if not args.no_intermediate_output:
        out_json_path = os.path.join(output_dir, 'SendRecvEvents_To_TaskCounter.json')
        with open(out_json_path, 'w') as json_file:
            json.dump(SendRecvEvents_To_TaskCounter, json_file, indent=4)
            json_file.write("\n\n")
        print('In-GPU goal file has been exported to InGPU_MicroEvents_Dependency.goal')

    # Goal_File_Name = './results/InterNode_MicroEvents_Dependency.goal'
    Goal_File_Name = os.path.join(output_dir, 'InterNode_MicroEvents_Dependency.goal')
    get_inter_node_microevents_dependency(Events_Parallel_Group, Comm_Init_Events, Comm_Info, SendRecvEvents_To_TaskCounter,
                                          Goal_File_Name, profile_interval, args.zero_red_copy, args.unique_nic)
    print('Internode goal file has been exported to InterNode_MicroEvents_Dependency.goal')

if __name__ == '__main__':
    main()