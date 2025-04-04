import re
import numpy as np
import matplotlib.pyplot as plt

# Change this to your log file path
log_file = "/home/tbonato/SC2025/atlahs/sim/htsim-backend/sim/datacenter/out.tmp"

# Regex to match the lines
pattern = re.compile(r"Flow\s+\S+\s+flowId\s+(\d+)\s+received SEND at\s+([\d\.]+)")

# Dictionary to store list of times per flow (using absolute time)
flow_times = {}

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            flow_id = match.group(1)
            t = float(match.group(2))  # time in microseconds
            flow_times.setdefault(flow_id, []).append(t)

# Set up common bins for throughput: here we use a bin size in microseconds
all_times = []
for times in flow_times.values():
    all_times.extend(times)
if not all_times:
    raise ValueError("No send events found in the file.")

tmin = min(all_times)
tmax = max(all_times)

# Define bin size in microseconds; change as needed
bin_size = 1.0  
bins = np.arange(tmin, tmax + bin_size, bin_size)

# Toggle filter: if enabled, only show the specified flow id.
FILTER_FLOW = True
FLOW_ID_TO_SHOW = "1000000771"

plt.figure(figsize=(10,6))
for fid, times in flow_times.items():
    if FILTER_FLOW and fid != FLOW_ID_TO_SHOW:
        continue

    # Bin the times; count events per bin
    counts, bin_edges = np.histogram(times, bins=bins)
    # Convert counts (events per bin) to throughput in Gbps.
    dt = bin_size * 1e-6  # bin interval in seconds
    # Each event is 4160 Bytes (packet size)
    throughput = (counts / dt) * (4160 * 8) / 1e9  # throughput in Gbps

    # Compute center of each time bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Plot from the flow's first packet to its last packet only
    first_time = min(times)
    last_time = max(times)
    mask = (bin_centers >= first_time) & (bin_centers <= last_time)
    plt.step(bin_centers[mask], throughput[mask], where="mid", label=f"Flow {fid}")

plt.xlabel("Absolute Time (μs)")
plt.ylabel("Throughput (Gbps)")
plt.title("Flow Throughput vs Time")
plt.legend()
plt.grid(True)
plt.show()