import json
import random
import numpy as np
from functools import lru_cache
from scipy import interpolate

# Used for storing the loaded data. Only be initialized once.
_DATA_CACHE = None

def init_data(simple_file, ll_file):
    """
    Load JSON files into the global variable _DATA_CACHE.
    """
    global _DATA_CACHE
    
    with open(simple_file, 'r') as f_simple:
        data_simple = json.load(f_simple)

    with open(ll_file, 'r') as f_ll:
        data_ll = json.load(f_ll)

    _DATA_CACHE = {
        '2': data_simple,  # Simple protocol
        '0': data_ll       # LL protocol
    }

@lru_cache(maxsize=2048)
def get_reduction_time(data_size, protocol, zero_red_copy=False):
    """
    data_size: The size of data (not including the flag).
    protocol:  '2' (Simple) or '0' (LL).
    """
    if zero_red_copy:
        return 0
    data = _DATA_CACHE[protocol]

    if str(data_size) in data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND']:
        reduction_times = data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(data_size)]
        return random.choice(reduction_times)

    # Use interpolation if the data_size is not directly in the JSON
    sizes = sorted(int(size) for size in data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'].keys())

    if data_size < sizes[0]:
        return random.choice(data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(sizes[0])])
    if data_size > sizes[-1]:
        return random.choice(data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(sizes[-1])])

    f = interpolate.interp1d(
        sizes,
        [np.mean(data['NPKIT_EVENT_GPU_RECV_REDUCE_SEND'][str(size)]) for size in sizes],
        kind='linear',
        fill_value="extrapolate"
    )
    interpolated_value = f(data_size)

    return int(random.gauss(interpolated_value, interpolated_value * 0.01))

@lru_cache(maxsize=2048)
def get_copy_time(data_size, protocol, zero_red_copy=False):
    """
    data_size: The size of data (not including the flag).
    protocol:  '2' (Simple) or '0' (LL).
    """
    if zero_red_copy:
        return 0
    data = _DATA_CACHE[protocol]

    if str(data_size) in data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND']:
        copy_times = data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(data_size)]
        return random.choice(copy_times)

    # Use interpolation if the data_size is not directly in the JSON
    sizes = sorted(int(size) for size in data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'].keys())

    if data_size < sizes[0]:
        return random.choice(data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(sizes[0])])
    if data_size > sizes[-1]:
        return random.choice(data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(sizes[-1])])

    f = interpolate.interp1d(
        sizes,
        [np.mean(data['NPKIT_EVENT_GPU_DIRECT_RECV_COPY_SEND'][str(size)]) for size in sizes],
        kind='linear',
        fill_value="extrapolate"
    )
    interpolated_value = f(data_size)

    return int(random.gauss(interpolated_value, interpolated_value * 0.01))
