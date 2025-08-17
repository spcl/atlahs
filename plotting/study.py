import os
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
import textwrap

def parse_file(filepath):
    """
    Parse the file and extract flow completion times (OT), drops (N) and 
    the collective completion time from lines like:
    F 148123 - ST 130887 - ET 133071 - OT 2183 - S 1048576 - N 0 - C 4096
    and a line like:
    It terminates! Htsim time 2309620922
    """
    flow_pattern = re.compile(r"F\s+\d+\s+-\s+ST\s+\d+\s+-\s+ET\s+\d+\s+-\s+OT\s+(\d+)\s+-\s+S\s+\d+\s+-\s+N\s+(\d+)\s+-\s+C\s+\d+")
    collective_pattern = re.compile(r"It terminates! Htsim time (\d+)")
    ot_list = []
    drops = 0
    collective_time = None
    with open(filepath, 'r') as f:
        for line in f:
            match_flow = flow_pattern.search(line)
            if match_flow:
                ot = int(match_flow.group(1))
                n = int(match_flow.group(2))
                ot_list.append(ot)
                drops += n
            match_collective = collective_pattern.search(line)
            if match_collective:
                collective_time = int(match_collective.group(1))
    return ot_list, drops, collective_time

def analyze_folder(folder):
    results = {}
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            ot_list, drops, collective_time = parse_file(filepath)
            if ot_list:
                max_ot = max(ot_list)
                avg_ot = sum(ot_list) / len(ot_list)
                percentile99 = np.percentile(ot_list, 99)
                percentile999 = np.percentile(ot_list, 99.9)
                results[filename] = {
                    "max_completion_time": max_ot,
                    "avg_completion_time": avg_ot,
                    "99_percentile_completion_time": percentile99,
                    "99.9_percentile_completion_time": percentile999,
                    "total_drops": drops,
                    "flow_count": len(ot_list),
                    "collective_completion_time": collective_time
                }
    return results

def split_label(label, width=10):
    return "\n".join(textwrap.wrap(label, width=width))

def plot_stats(stats):
    files = list(stats.keys())
    # Remove specific substring from labels and wrap them to multiple lines.
    multi_labels = [split_label(f.replace("ATLAHS_llama_N16_GPU64_1iter.bin", ""), width=13) for f in files]
    
    max_ct = [stats[f]["max_completion_time"] for f in files]
    avg_ct = [stats[f]["avg_completion_time"] for f in files]
    p99_ct = [stats[f]["99_percentile_completion_time"] for f in files]
    p999_ct = [stats[f]["99.9_percentile_completion_time"] for f in files]
    drops = [stats[f]["total_drops"] for f in files]
    collective = [stats[f]["collective_completion_time"] for f in files]
    
    x = range(len(files))
    
    # Create a 3x2 grid for six plots
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))
    
    # Max Completion Time
    bars = axs[0,0].bar(x, max_ct, color='skyblue')
    axs[0,0].set_title("Max Completion Time (us)")
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(multi_labels, rotation=0, fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axs[0,0].text(bar.get_x() + bar.get_width()/2, height, f'{max_ct[i]}',
                      ha='center', va='bottom', fontsize=7)
    
    # Average Completion Time
    bars = axs[0,1].bar(x, avg_ct, color='lightgreen')
    axs[0,1].set_title("Average Completion Time (us)")
    axs[0,1].set_xticks(x)
    axs[0,1].set_xticklabels(multi_labels, rotation=0, fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axs[0,1].text(bar.get_x() + bar.get_width()/2, height, f'{avg_ct[i]:.2f}',
                      ha='center', va='bottom', fontsize=7)
    
    # 99% Percentile Completion Time
    bars = axs[1,0].bar(x, p99_ct, color='salmon')
    axs[1,0].set_title("99% Percentile Completion Time (us)")
    axs[1,0].set_xticks(x)
    axs[1,0].set_xticklabels(multi_labels, rotation=0, fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axs[1,0].text(bar.get_x() + bar.get_width()/2, height, f'{p99_ct[i]}',
                      ha='center', va='bottom', fontsize=7)
    
    # Total Drops
    bars = axs[1,1].bar(x, drops, color='violet')
    axs[1,1].set_title("Total Drops")
    axs[1,1].set_xticks(x)
    axs[1,1].set_xticklabels(multi_labels, rotation=0, fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axs[1,1].text(bar.get_x() + bar.get_width()/2, height, f'{drops[i]}',
                      ha='center', va='bottom', fontsize=7)
    
    # Collective Completion Time
    bars = axs[2,0].bar(x, collective, color='gold')
    axs[2,0].set_title("Collective Completion Time (us)")
    axs[2,0].set_xticks(x)
    axs[2,0].set_xticklabels(multi_labels, rotation=0, fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{collective[i]}' if collective[i] is not None else "-"
        axs[2,0].text(bar.get_x() + bar.get_width()/2, height, label,
                      ha='center', va='bottom', fontsize=7)
    
    # 99.9% Percentile Completion Time
    bars = axs[2,1].bar(x, p999_ct, color='orchid')
    axs[2,1].set_title("99.9% Percentile Completion Time (us)")
    axs[2,1].set_xticks(x)
    axs[2,1].set_xticklabels(multi_labels, rotation=0, fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axs[2,1].text(bar.get_x() + bar.get_width()/2, height, f'{p999_ct[i]}',
                      ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze flow completion times from files in a folder")
    parser.add_argument("--folder", help="Folder containing the flow output files", required=True)
    args = parser.parse_args()
    
    stats = analyze_folder(args.folder)
    for filename, data in stats.items():
        print(f"File: {filename}")
        print(f"  Max completion time: {data['max_completion_time']} us")
        print(f"  Average completion time: {data['avg_completion_time']:.2f} us")
        print(f"  99% percentile completion time: {data['99_percentile_completion_time']} us")
        print(f"  99.9% percentile completion time: {data['99.9_percentile_completion_time']} us")
        print(f"  Total drops: {data['total_drops']}")
        print(f"  Total flows: {data['flow_count']}")
        if data['collective_completion_time'] is not None:
            print(f"  Collective completion time: {data['collective_completion_time']} us")
        print("")
    
    plot_stats(stats)