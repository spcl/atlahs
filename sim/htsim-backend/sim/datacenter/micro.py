import os

os.system("./htsim_uec -topo topologies/leaf_spine_64_1os.topo -tm connection_matrices/incast.cm -linkspeed 200000 -nodes 64 -strat ecmp_host -mtu 4096 -paths 128 -q 200000 -algorithm swift_like > ../../../../plotting/micro/swift_incast.tmp")
os.system("./htsim_uec -topo topologies/leaf_spine_64_1os.topo -tm connection_matrices/incast.cm -linkspeed 200000 -nodes 64 -strat ecmp_host -mtu 4096 -paths 128 -q 200000 -algorithm mprdma > ../../../../plotting/micro/mprdma_incast.tmp")
os.system("./htsim_uec -topo topologies/leaf_spine_64_1os.topo -tm connection_matrices/perm_32.cm -linkspeed 200000 -nodes 64 -strat ecmp_host -mtu 4096 -paths 128 -q 200000 -algorithm swift_like > ../../../../plotting/micro/swift_perm.tmp")
os.system("./htsim_uec -topo topologies/leaf_spine_64_1os.topo -tm connection_matrices/perm_32.cm -linkspeed 200000 -nodes 64 -strat ecmp_host -mtu 4096 -paths 128 -q 200000 -algorithm mprdma > ../../../../plotting/micro/mprdma_perm.tmp")
