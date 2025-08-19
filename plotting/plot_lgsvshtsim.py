import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Define groups and algorithm labels
groups = ["No Oversubscription", "4:1 Oversubscription"]
algorithms = ["ATLAHS LGS", "ATLAHS htsim"]



# Dummy data for total run time for each algorithm and group
run_time_data = {
    "ATLAHS LGS": [2.12, 2.12],      # Dummy values for each group
    "ATLAHS htsim": [2.13, 4.61]     # Dummy values for each group
}

dropped_packets_data = [292765, 206409686]  

# Setup the x locations for the groups
x = np.arange(len(groups))
width = 0.35  # bar width

# Create subplots: left for run time, right for dropped packets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_axisbelow(True)  # Ensures grid lines appear behind the bars in ax1
ax2.set_axisbelow(True)  # Ensures grid lines appear behind the bars in ax2

# Left plot: grouped bar chart for Total Run Time
ax1.bar(x - width/2, run_time_data["ATLAHS LGS"], width, label="ATLAHS LGS", color="#20b2aa")
ax1.bar(x + width/2, run_time_data["ATLAHS htsim"], width, label="ATLAHS htsim", color="#40e0d0")
ax1.set_xticks(x)
ax1.set_xticklabels(groups, fontsize=14)  # increased tick label size
ax1.set_ylabel("Time For Training Iteration (s)", fontsize=14.5)
ax1.set_xlabel("Topology Configuration", fontsize=14.5)
ax1.legend(prop={'size': 14})

# Add horizontal dotted grid lines
ax1.yaxis.grid(True, linestyle=':', color='black', alpha=0.7)

# Annotate percentage difference on top of the ATLAHS LGS bars
for i in range(len(groups)):
    lgs_val = run_time_data["ATLAHS LGS"][i]
    htsim_val = run_time_data["ATLAHS htsim"][i]
    diff_percentage = ((lgs_val - htsim_val) / lgs_val) * 100 if lgs_val != 0 else 0
    annotation = f"{diff_percentage:+.1f}%"
    # Offset the annotation slightly above the LGS bar
    ax1.text(x[i] - width/2, lgs_val + 0.05 * lgs_val, annotation, ha="center", va="bottom", color="black", fontsize=11.5)

# Right plot: bar chart for Total Dropped Packets (only ATLAHS htsim)
right_width = 0.5  # Increased bar width to reduce spacing between bars
ax2.bar(x, dropped_packets_data, right_width, color="#40e0d0")
ax2.set_xticks(x)
ax2.set_xticklabels(groups, fontsize=14)
ax2.set_ylabel("Total Packet Drops", fontsize=14.5)
ax2.set_xlabel("Topology Configuration", fontsize=14.5)

# Add horizontal dotted grid lines
ax2.yaxis.grid(True, linestyle=':', color='black', alpha=0.7)

plt.tight_layout()
plt.savefig("lgs_vs_htsim.pdf", dpi=300)
plt.show()