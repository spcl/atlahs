#!/usr/bin/env python3
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# Packet size in bytes
PACKET_SIZE = 4096

def main(filename):
    flows = {}  # Key: flow identifier, Value: list of timestamps
    pattern = re.compile(r'Flow\s+(\S+)\s+-\s+Send\s+(\d+)')
    with open(filename, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                flow_id = m.group(1)
                ts = int(m.group(2))
                flows.setdefault(flow_id, []).append(ts)
                
    if not flows:
        print("No valid data found in the file.")
        return

    plt.figure(figsize=(10, 5))
    for flow_id, timestamps in flows.items():
        timestamps.sort()
        orig_start = timestamps[0]
        # Ignore timestamps in the first 10 microseconds (10,000 ns) of data
        threshold = orig_start + 10_000  # 10 microseconds in ns
        timestamps = [t for t in timestamps if t >= threshold]
        if not timestamps:
            continue
            
        start = timestamps[0]
        end = timestamps[-1]
    
        # Define bin width (1e3 ns as chosen)
        bin_width = int(1e4)
        bins = np.arange(start, end + bin_width, bin_width)
        counts, edges = np.histogram(timestamps, bins=bins)
    
        # Throughput in bytes per second (bin width in ns to seconds)
        throughput = counts * PACKET_SIZE
    
        # Compute the time (in seconds) for each bin (relative to start)
        t = (edges[:-1] - start) * 1e-9  # convert ns to seconds
    
        plt.plot(t, throughput, drawstyle='steps-post', marker='o', label=flow_id)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Bytes/s)")
    plt.title("Throughput Over Time per Flow")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_throughput.py <data_file>")
        sys.exit(1)
    main(sys.argv[1])