import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Modify the path below if needed
logfile = '/home/tbonato/SC2025/atlahs/sim/htsim-backend/sim/datacenter/swift_incast.tmp'

# Dictionary to store CWND values per flow.
# Each key will map to a list of (index, cwnd) pairs.
flow_cwnd = defaultdict(list)

# Regular expression to capture lines with "- CWND" info.
# Assumes lines like: "Flow uec_10_14  - CWND 68888"
pattern = re.compile(r'Flow\s+(\S+)\s+-\s+CWND\s+(\d+)')

with open(logfile, 'r') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            flow = m.group(1)
            cwnd = int(m.group(2))
            # Use the current count for this flow as a time/index marker.
            flow_cwnd[flow].append(cwnd)

# Create a plot for the flows.
plt.figure(figsize=(12, 8))
for flow, cwnd_values in flow_cwnd.items():
    plt.plot(range(len(cwnd_values)), cwnd_values, label=flow)

plt.xlabel('Event Number')
plt.ylabel('CWND (bytes)')
plt.title('CWND Evolution per Flow')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()