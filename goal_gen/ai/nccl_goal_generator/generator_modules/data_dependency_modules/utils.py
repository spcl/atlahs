def modRanks(r, nranks):
    return r - nranks if r >= nranks else r     

def div_up(x, y):
    return (x + y - 1) // y

def get_event_type(operation):
    if operation == 'AllReduce':
        return 0
    elif operation == 'Broadcast':
        return 1
    elif operation == 'AllGather':
        return 2
    if operation == 'ReduceScatter':
        return 3
    if operation == 'Reduce':
        return 4
    elif operation == 'Send':
        return 5
    elif operation == 'Recv':
        return 5