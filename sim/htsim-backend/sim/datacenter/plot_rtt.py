import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Update the path to your log file if different
log_file = 'swift_incast.tmp'

# Pattern to capture lines like: "Flow uec_0_15  - RTT 9351"
pattern = re.compile(r'Flow (\S+)\s*-\s*RTT (\d+)')

# Dictionary: flow -> list of (event_index, rtt)
data = defaultdict(list)
event_index = 0

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            flow = match.group(1)
            rtt = int(match.group(2))
            data[flow].append((event_index, rtt))
            event_index += 1

plt.figure(figsize=(10, 6))
for flow, points in data.items():
    if points:
        times, rtts = zip(*points)
        plt.plot(times, rtts, label=flow)

plt.xlabel('Event Index (time)')
plt.ylabel('RTT')
plt.title('RTT Over Time for Each Flow')
plt.legend()
plt.tight_layout()
plt.show()