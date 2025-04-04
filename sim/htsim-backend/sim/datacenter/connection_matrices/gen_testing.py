# Define parameters
num_connections = 5                # Sender nodes 0 through 10 (11 connections)
receiver = 33                       # Fixed receiver node (can be changed)
size_bytes = 1 * 1024 * 1024          # 1 MiB in bytes
delay_ns = 62004                    # Delay in nanoseconds between connections
delay_ps = delay_ns * 1000          # Convert ns to ps

# Create the list of connection strings
connections = []
start_time = 0
num_id = 1
for sender in range(num_connections):  # senders 0, 1, ... 10
    line = f"0->11 id {num_id} start {start_time} size {size_bytes}"
    connections.append(line)
    start_time += delay_ps  # Increment start time by the delay (in picoseconds)
    num_id += 1

# Determine the total number of nodes
# (Assuming nodes are numbered from 0 up to the maximum index seen)
# Here we use the maximum of the sender range and the receiver.
num_nodes = max(num_connections - 1, receiver) + 1

# Output the connection matrix
print(f"Nodes 32")
print(f"Connections {len(connections)}")
for conn in connections:
    print(conn)
