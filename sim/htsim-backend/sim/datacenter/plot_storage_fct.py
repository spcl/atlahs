import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def read_overall_times(filename):
    times = []
    regex = re.compile(r"OT\s+(\d+)")
    with open(filename, 'r') as f:
        for line in f:
            match = regex.search(line)
            if match:
                # Divide by 1000 to convert to microseconds.
                times.append(int(match.group(1)) / 1000)
    return times

def plot_four_violins(data1, data2, data3, data4,
                      label1="MPRDMA", label2="NDP"):
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Define positions for the four violins with less distance between groups.
    positions = [1, 2, 3, 4]
    violin_data = [data1, data2, data3, data4]
    
    # Create the violins without medians/extrema.
    vp = ax.violinplot(violin_data, positions=positions,
                         showmedians=False, showmeans=False, showextrema=False)
    
    # Customize colors for the violins: use lightcoral for MPRDMA and palegreen for NDP.
    colors = ['lightcoral', 'palegreen', 'lightcoral', 'palegreen']
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    
    # Compute max, 99th percentile and mean for each dataset.
    max_vals = [np.max(d) for d in violin_data]
    p99_vals = [np.percentile(d, 99) for d in violin_data]
    means   = [np.mean(d) for d in violin_data]
    
    # Overlay scatter markers (using s=20 for slightly smaller markers)
    # Group 1: positions 1 and 2.
    ax.scatter([positions[0]], [max_vals[0]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[0]], [p99_vals[0]], color='mediumseagreen', marker='x', s=20, zorder=3)
    ax.scatter([positions[1]], [max_vals[1]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[1]], [p99_vals[1]], color='mediumseagreen', marker='x', s=20, zorder=3)
    # Group 2: positions 3 and 4.
    ax.scatter([positions[2]], [max_vals[2]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[2]], [p99_vals[2]], color='mediumseagreen', marker='x', s=20, zorder=3)
    ax.scatter([positions[3]], [max_vals[3]], color='firebrick', marker='o', s=20, zorder=3)
    ax.scatter([positions[3]], [p99_vals[3]], color='mediumseagreen', marker='x', s=20, zorder=3)
    
    # Add extra scatter markers for the mean (dodgerblue).
    for pos, mean in zip(positions, means):
        ax.scatter([pos], [mean], color='dodgerblue', marker='o', s=20, zorder=3)
    
    # Annotate each violin with max, 99th percentile, and mean values (fontsize increased).
    for i, pos in enumerate(positions):
        ax.text(pos + 0.05, max_vals[i], f"{max_vals[i]:.1f} µs",
                color='firebrick', fontsize=11, verticalalignment='center')
        ax.text(pos + 0.05, p99_vals[i], f"{p99_vals[i]:.1f} µs",
                color='mediumseagreen', fontsize=11, verticalalignment='center')
        ax.text(pos + 0.05, means[i] + 0.2, f"{means[i]:.1f} µs",
                color='dodgerblue', fontsize=11, verticalalignment='center')
    
    # Set x-axis ticks and labels (fontsize increased in tick_params).
    ax.set_xticks(positions)
    ax.set_xticklabels([label1, label2, label1, label2], fontsize=11)
    ax.set_ylabel("MCT (µs)", fontsize=13)
    
    # Use only horizontal grid lines.
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    
    # Custom legend for markers (with smaller marker size and increased font size in legend).
    custom_handles = [
        Line2D([], [], color='firebrick', marker='o', linestyle='None', markersize=6, label='Max'),
        Line2D([], [], color='mediumseagreen', marker='o', linestyle='None', markersize=6, label='99th Percentile'),
        Line2D([], [], color='dodgerblue', marker='o', linestyle='None', markersize=6, label='Mean')
    ]
    ax.legend(handles=custom_handles, fontsize=11)
    
    # Add oversubscription group labels below the x-axis labels (plain text, fontsize increased).
    ax.text(1.5, -0.16, "No Oversubscription", ha='center', va='center',
            transform=ax.get_xaxis_transform(), fontsize=13)
    ax.text(3.5, -0.16, "8:1 Oversubscription", ha='center', va='center',
            transform=ax.get_xaxis_transform(), fontsize=13)
    
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig("flow_completion_times_storage.pdf", dpi=300)
    plt.show()

def main():
    # Filenames for the two groups.

    filename1 = "1os_mprdma.tmp"       # Group 1, MPRDMA (no oversubscription)
    filename2 = "1os_eqds.tmp"     # Group 1, NDP (no 4:1)
    filename3 = "8os_mprdma.tmp"       # Group 2, MPRDMA (4:1 oversubscription)
    filename4 = "8os_eqds.tmp"     # Group 2, NDP (4:1 oversubscription)

    """ os.system(f"./htsim_uec -lgs_flow_stats -seed 4 -topo topologies/leaf_spine_128_1os.topo -goal ../lgs/input/storage.bin  -linkspeed 200000 -nodes 1024 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {filename1}")
    os.system(f"./htsim_ndp -lgs_flow_stats -seed 4 -topo topologies/leaf_spine_128_1os.topo -goal ../lgs/input/storage.bin  -linkspeed 200000 -nodes 1024 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {filename2}")
    os.system(f"./htsim_uec -lgs_flow_stats -seed 4 -topo topologies/leaf_spine_128_8os.topo -goal ../lgs/input/storage.bin  -linkspeed 200000 -nodes 1024 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {filename3}")
    os.system(f"./htsim_ndp -lgs_flow_stats -seed 4 -topo topologies/leaf_spine_128_8os.topo -goal ../lgs/input/storage.bin  -linkspeed 200000 -nodes 1024 -strat ecmp_host -mtu 4096 -paths 128 -q 1000000 > {filename4}") """

    data1 = read_overall_times(filename1)
    data2 = read_overall_times(filename2)
    data3 = read_overall_times(filename3)
    data4 = read_overall_times(filename4)
    
    missing = False
    for f, d in [(filename1, data1), (filename2, data2),
                 (filename3, data3), (filename4, data4)]:
        if not d:
            print(f"No overall time data found in {f}.")
            missing = True
            
    if not missing:
        plot_four_violins(data1, data2, data3, data4,
                          label1="MPRDMA", label2="NDP")

if __name__ == "__main__":
    main()