import matplotlib.pyplot as plt
import numpy as np

#os.system("./htsim_uec -topo topologies/leaf_spine_128_4os.topo -goal ../lgs/input/merged_random.bin  -linkspeed 200000 -nodes 1024 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > res_random.tmp")
#os.system("./htsim_uec -topo topologies/leaf_spine_128_4os.topo -goal ../lgs/input/merged.bin  -linkspeed 200000 -nodes 1024 -strat ecmp_host -mtu 4096 -paths 128 -lgs_flow_stats -q 1000000 > res_packed.tmp")
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Dummy data: runtime in seconds for each job under two allocation types
runtime_data = {
    'Packed Allocation': {'Llama ': 2.11, 'LULESH': 5.44},
    'Random Allocation': {'Llama ': 2.85, 'LULESH': 5.48},
}

# Extract groups and job labels
groups = list(runtime_data.keys())          # Should be ['Packed Allocation', 'Random Allocation']
jobs = list(next(iter(runtime_data.values())).keys())

n_groups = len(groups)
n_jobs   = len(jobs)

bar_width = 0.35
index = np.arange(n_groups)

# Create figure with a more horizontal layout
fig, ax = plt.subplots(figsize=(7, 3))

# Ensure the grid lines appear below the bar chart and only horizontal grid lines are shown
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--')

# Create a bar for each job with appropriate offsets
for i, job in enumerate(jobs):
    # Determine bar color based on the job name (trimmed)
    job_clean = job.strip()
    if job_clean == 'Llama':
        color = "#2eaba9"
    elif job_clean == 'LULESH':
        color = "#fb8574"
    else:
        color = None  # Use default matplotlib color if no match

    # Shift bars by their index so they appear side by side
    positions = index + (i * bar_width) - ((n_jobs - 1) * bar_width / 2)
    values = [runtime_data[group][job] for group in groups]
    ax.bar(positions, values, bar_width, label=job, color=color, zorder=2)

    # Annotate Random Allocation bar (assumed to be the second group) with % increase
    if len(values) >= 2:
        packed_val = runtime_data[groups[0]][job]
        random_val = runtime_data[groups[1]][job]
        percent_increase = (random_val - packed_val) / packed_val * 100
        annotate_text = f"+{percent_increase:.0f}%"
        # The x coordinate for the Random Allocation bar is the second element in positions
        ax.text(positions[1], random_val + 0.2, annotate_text, color='red',
                ha='center', va='bottom', fontsize=12)

ax.set_xlabel('Allocation Type', fontsize=14)
ax.set_ylabel('Simulated Runtime (s)', fontsize=14)
#ax.set_title('Runtime by Allocation Type and Job', fontsize=16)
ax.set_xticks(index)
ax.set_ylim(0, 7)
ax.set_xticklabels(groups, fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig("job_allocation_runtime.pdf", dpi=300)
plt.savefig("job_allocation_runtime.png", dpi=300)
plt.show()