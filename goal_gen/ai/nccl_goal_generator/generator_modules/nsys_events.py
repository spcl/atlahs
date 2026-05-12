import argparse
import yaml
import os
import glob
import json
import math
import sqlite3
import re
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from queue import Queue

#### Postprocessing nsys files
def _parse_nccl_debug_topology_from_logs(log_glob: str) -> Dict[str, Dict[str, object]]:
    """
    Best-effort parser for NCCL DEBUG/INFO logs to reconstruct init/topology metadata
    when NVTX INIT markers are missing from the trace (e.g., due to late tracing).

    Expected (substrings) in logs (format varies by NCCL version/build).
    Two common styles are supported:

    (A) NVTX-like debug lines (some patched builds):
      - commHash <H> commId <C> rank <r> nranks <n>
      - commHash <H> Rings [<ch>] <prev>-><me>-><next>
      - commHash <H> Trees [<ch>] <c1>/<c2>/<c3>-><me>-><parent>

    (B) Standard NCCL INFO lines:
      - ncclCommInitRank... rank <r> nranks <n> ... commId <C> - Init START/COMPLETE
      - Ring <ch> : <prev> -> <me> -> <next>
      - Trees [<ch>] <c1>/<c2>/<c3>-><me>-><parent>

    Returns:
      topo_by_id[commId_or_commHash] = {
        'nranks': int,
        # Optional mapping only available for standard NCCL INFO logs when the
        # log filename encodes host+pid (e.g., NCCL_DEBUG_FILE=".../nccl_%h_%p.log").
        # Keys are (host, pid) tuples with pid as int.
        'pid_to_rank': {(host_str, pid_int): rank_int},
        'ring': {(rank_str, ch_str): {'previous_rank': str, 'my_rank': str, 'next_rank': str, 'channel_Id': str}},
        'tree': {(rank_str, ch_str): {'child_1_rank': str, 'child_2_rank': str, 'child_3_rank': str, 'my_rank': str, 'parent_rank': str, 'channel_Id': str}},
      }
    """
    if not log_glob:
        return {}

    # Expand glob and keep only regular files.
    paths = [p for p in glob.glob(log_glob) if os.path.isfile(p)]
    if len(paths) == 0:
        return {}

    # (A) NVTX-like patterns.
    re_comm_a = re.compile(r"commHash (\S+) commId (\S+) rank (\d+) nranks (\d+)")
    re_ring_a = re.compile(r"commHash (\S+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+)")
    re_tree_a = re.compile(r"commHash (\S+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+)")

    # (B) Standard NCCL INFO patterns.
    re_init_b = re.compile(r"\brank (\d+)\s+nranks (\d+).*\bcommId (0x[0-9a-fA-F]+)\b.*- Init (START|COMPLETE)")
    re_ring_b = re.compile(r"\bRing\s+(\d+)\s*:\s*(\d+)\s*->\s*(\d+)\s*->\s*(\d+)")
    re_tree_b = re.compile(r"\bTrees\s*\[(\d+)\]\s+(-?\d+)/(-?\d+)/(-?\d+)->\s*(-?\d+)->\s*(-?\d+)")
    # Some NCCL versions print multiple channel entries on the same line, e.g.:
    #   "Trees [0] ... [1] ... [2] ..."
    # Parse all segments instead of only the first one.
    re_tree_seg_b = re.compile(r"\[(\d+)\]\s+(-?\d+)/(-?\d+)/(-?\d+)->\s*(-?\d+)->\s*(-?\d+)")

    topo: Dict[str, Dict[str, object]] = {}

    for path in sorted(paths):
        try:
            # Infer host/pid from standard NCCL_DEBUG_FILE naming.
            file_host = None
            file_pid = None
            mfn = re.search(r"nccl_([^_/]+)_(\d+)\.log$", os.path.basename(path))
            if mfn:
                file_host = mfn.group(1)
                try:
                    file_pid = int(mfn.group(2))
                except Exception:
                    file_pid = None

            with open(path, "r", errors="replace") as f:
                active_id = None
                for line in f:
                    # (A) NVTX-like.
                    if "commHash" in line:
                        m = re_comm_a.search(line)
                        if m:
                            commHash = m.group(1)
                            nranks = int(m.group(4))
                            entry = topo.setdefault(commHash, {"nranks": nranks, "ring": {}, "tree": {}})
                            # Prefer the largest nranks observed (some logs may be truncated).
                            try:
                                entry["nranks"] = max(int(entry.get("nranks", 0)), nranks)
                            except Exception:
                                entry["nranks"] = nranks
                            continue

                        m = re_ring_a.search(line)
                        if m:
                            commHash = m.group(1)
                            ch = m.group(2)
                            prev_r = m.group(3)
                            me_r = m.group(4)
                            next_r = m.group(5)
                            entry = topo.setdefault(commHash, {"nranks": None, "ring": {}, "tree": {}})
                            entry["ring"][(me_r, ch)] = {
                                "previous_rank": prev_r,
                                "my_rank": me_r,
                                "next_rank": next_r,
                                "channel_Id": ch,
                            }
                            continue

                        m = re_tree_a.search(line)
                        if m:
                            commHash = m.group(1)
                            entry = topo.setdefault(commHash, {"nranks": None, "ring": {}, "tree": {}})
                            # Prefer parsing all channel segments on the line (some NCCL prints
                            # multiple channels after "Trees [0] ...").
                            for ch, c1, c2, c3, me_r, parent in re_tree_seg_b.findall(line):
                                entry["tree"][(me_r, ch)] = {
                                    "child_1_rank": c1,
                                    "child_2_rank": c2,
                                    "child_3_rank": c3,
                                    "my_rank": me_r,
                                    "parent_rank": parent,
                                    "channel_Id": ch,
                                }
                            continue

                    # (B) Standard NCCL INFO.
                    if "commId" in line and "Init" in line:
                        m = re_init_b.search(line)
                        if m:
                            # Key by commId since that's the only stable identifier in standard logs.
                            active_id = m.group(3)
                            nranks = int(m.group(2))
                            entry = topo.setdefault(active_id, {"nranks": nranks, "pid_to_rank": {}, "ring": {}, "tree": {}})
                            try:
                                entry["nranks"] = max(int(entry.get("nranks", 0)), nranks)
                            except Exception:
                                entry["nranks"] = nranks
                            # Record pid->rank mapping if available.
                            if file_host is not None and file_pid is not None:
                                try:
                                    entry.setdefault("pid_to_rank", {})[(file_host, file_pid)] = int(m.group(1))
                                except Exception:
                                    pass
                            # Do not clear active_id on "Init COMPLETE": some NCCL versions print
                            # Ring/Tree topology after the COMPLETE line.
                            continue

                    if active_id is not None:
                        if "Ring" in line:
                            m = re_ring_b.search(line)
                            if m:
                                ch = m.group(1)
                                prev_r = m.group(2)
                                me_r = m.group(3)
                                next_r = m.group(4)
                                entry = topo.setdefault(active_id, {"nranks": None, "ring": {}, "tree": {}})
                                entry.setdefault("pid_to_rank", {})
                                entry["ring"][(me_r, ch)] = {
                                    "previous_rank": prev_r,
                                    "my_rank": me_r,
                                    "next_rank": next_r,
                                    "channel_Id": ch,
                                }
                                continue

                        if "Trees" in line:
                            segs = re_tree_seg_b.findall(line)
                            if segs:
                                entry = topo.setdefault(active_id, {"nranks": None, "ring": {}, "tree": {}})
                                entry.setdefault("pid_to_rank", {})
                                for ch, c1, c2, c3, me_r, parent in segs:
                                    entry["tree"][(me_r, ch)] = {
                                        "child_1_rank": c1,
                                        "child_2_rank": c2,
                                        "child_3_rank": c3,
                                        "my_rank": me_r,
                                        "parent_rank": parent,
                                        "channel_Id": ch,
                                    }
                                continue
        except OSError:
            continue

    # Drop entries without any ring/tree info.
    topo = {h: v for h, v in topo.items() if len(v.get("ring", {})) > 0 or len(v.get("tree", {})) > 0}
    return topo


def get_nsys_events(dir_path, nccl_debug_log_glob: Optional[str] = None, use_nccl_debug_topology: bool = False):
    comm_info = {}
    nccl_events = {}
    profile_interval = {}
    cupti_kernel_results = {}
    HostName_To_GoalRank = {}
    GoalRank_To_NumOfGPUs = {}
    commHash_to_commId = {}
    stream_to_streamId = {}
    comm_init_events = {}
    events_counter = {}
    ts_group_start= {}
    ts_group_end = {}
    gpuId = -1
    known_gpus = -1
    file_count = 0
    profile_start_slack_ns = int(os.environ.get("ATLAHS_PROFILE_START_SLACK_NS", "0").strip() or "0")
    profile_end_slack_ns = int(os.environ.get("ATLAHS_PROFILE_END_SLACK_NS", "0").strip() or "0")
    file_names = os.listdir(dir_path)
    if os.environ.get("ATLAHS_SORT_NSYS_FILES", "1").strip() != "0":
        file_names = sorted(file_names)

    # Optional: pre-scan all sqlite files to compute a single global (intersection) profiling
    # window. This is particularly helpful for late-capture traces where each process may call
    # cudaProfilerStart/Stop at slightly different times, producing per-node windows that do not
    # align and inflate simulated makespans.
    global_profile_window = None  # (start_ns, end_ns)
    use_global_intersection = os.environ.get("ATLAHS_PROFILE_USE_GLOBAL_INTERSECTION", "0").strip() == "1"
    if use_global_intersection:
        starts = []
        ends = []
        for fn in file_names:
            if not fn.endswith(".sqlite"):
                continue
            p = os.path.join(dir_path, fn)
            try:
                conn = sqlite3.connect(p)
                cursor = conn.cursor()

                # Prefer NVTX markers (patched Megatron).
                cursor.execute("SELECT text, start FROM NVTX_EVENTS WHERE text LIKE 'nsys profiling%'")
                rows = cursor.fetchall()
                s = None
                e = None
                if rows:
                    for txt, ts in rows:
                        if not txt:
                            continue
                        if "nsys profiling start" in txt:
                            s = ts if s is None else min(s, ts)
                        elif "nsys profiling stopped" in txt:
                            e = ts if e is None else max(e, ts)

                # Fallback: DIAGNOSTIC_EVENT messages (capture-range=cudaProfilerApi).
                if s is None or e is None:
                    try:
                        cursor.execute(
                            "SELECT timestamp, text FROM DIAGNOSTIC_EVENT "
                            "WHERE text LIKE 'Profiling started due to cudaProfilerStart%' "
                            "   OR text LIKE 'Profiling has started%' "
                            "   OR text LIKE 'Profiling has stopped%' "
                            "ORDER BY timestamp"
                        )
                        drows = cursor.fetchall()
                        ds = None
                        de = None
                        for ts, txt in drows:
                            if not txt:
                                continue
                            if txt.startswith("Profiling started due to cudaProfilerStart") or txt.startswith("Profiling has started"):
                                ds = ts if ds is None else min(ds, ts)
                            elif txt.startswith("Profiling has stopped"):
                                de = ts if de is None else max(de, ts)
                        if ds is not None and de is not None:
                            # Prefer cudaProfilerStart-derived start if present.
                            s = ds if s is None else min(s, ds)
                            e = de if e is None else max(e, de)
                    except Exception:
                        pass

                conn.close()
                if s is None or e is None:
                    continue
                if int(e) > int(s):
                    starts.append(int(s))
                    ends.append(int(e))
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                continue

        if len(starts) > 0 and len(ends) > 0:
            gstart = max(starts)
            gend = min(ends)
            if gend > gstart:
                global_profile_window = (int(gstart), int(gend))
                print(f"[INFO] Global intersection profile window: [{gstart}, {gend}] ({(gend-gstart)/1e9:.6f} s)")

    # NCCL 2.28 traces may be exported as one sqlite per rank with names like:
    #   profile_<jobid>_<node>_<rank>.sqlite
    # In that case, comm init NVTX markers might be missing (profiling started late),
    # but we can still infer:
    # - nranks from filenames
    # - host<->goal_rank mapping from Hostname in TARGET_INFO_SYSTEM_ENV (best effort)
    inferred_rank_to_host = {}
    inferred_rank_to_node = {}
    profile_rank_re = re.compile(r"profile_(\d+)_(\d+)_(\d+)\.sqlite$")
    inferred_ranks = set()
    for fn in file_names:
        if not fn.endswith(".sqlite"):
            continue
        m = profile_rank_re.match(fn)
        if not m:
            continue
        try:
            node = int(m.group(2))
            rank = int(m.group(3))
        except ValueError:
            continue
        inferred_ranks.add(rank)
        inferred_rank_to_node[rank] = node
        file_path = os.path.join(dir_path, fn)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            continue
        try:
            con = sqlite3.connect(file_path)
            cur = con.cursor()
            cur.execute("select value from TARGET_INFO_SYSTEM_ENV where name='Hostname' limit 1")
            row = cur.fetchone()
            if row and row[0]:
                inferred_rank_to_host[rank] = row[0]
        except Exception:
            pass
        finally:
            try:
                con.close()
            except Exception:
                pass

    inferred_nranks = (max(inferred_ranks) + 1) if len(inferred_ranks) > 0 else None
    if inferred_nranks is not None:
        host_to_min_rank = defaultdict(lambda: 10**18)
        for r in range(inferred_nranks):
            hn = inferred_rank_to_host.get(r)
            if not hn:
                node = inferred_rank_to_node.get(r, 0)
                hn = f"node{node}"
            host_to_min_rank[hn] = min(host_to_min_rank[hn], int(r))
        host_order = sorted(host_to_min_rank.keys(), key=lambda hn: host_to_min_rank[hn])
        HostName_To_GoalRank = {hn: i for i, hn in enumerate(host_order)}
        GoalRank_To_NumOfGPUs = {i: 0 for i in range(len(host_order))}

    # Cross-file mapping (per-rank traces): global rank -> synthetic gpuId.
    # Filled lazily when we see NCCL-relevant PIDs in each sqlite.
    global_rank_to_gpuId = {}
    # For per-node traces, we can still map per-node PIDs to gpuId. This is
    # required to attach NCCL-debug-log rank mappings to the right gpuId.
    hostpid_to_gpuId: Dict[Tuple[str, int], int] = {}
    gpuId_to_hostpid: Dict[int, Tuple[str, int]] = {}
    commId_to_hostpids: Dict[str, set] = defaultdict(set)

    for file_name in file_names:  ## each file may represent a host(root process), containing info of all GPUs (one GPU per child process) or a process corresponding to one GPU
        if file_name.endswith('.sqlite'):
            file_path = os.path.join(dir_path, file_name)
            if os.path.getsize(file_path) == 0:
                continue

            file_count += 1
            
            pid_to_gpuId = {}

            Parse_State = {}
            last_Coll_streamId = {}
            last_P2P_streamId = {}
            last_update = {}

            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            # Hostname resolution:
            # - 2.20 per-node files: nsys_report_<hostname>....sqlite
            # - 2.28 per-rank files: profile_<job>_<node>_<rank>.sqlite (hostname lives in TARGET_INFO_SYSTEM_ENV)
            inferred_rank = None
            m = profile_rank_re.match(file_name)
            if m:
                try:
                    inferred_rank = int(m.group(3))
                except ValueError:
                    inferred_rank = None

            host_name = None
            match = re.search(r'nsys_report_([^.]+)\\.', file_name)
            if match:
                host_name = match.group(1)
            elif inferred_rank is not None and inferred_rank in inferred_rank_to_host:
                host_name = inferred_rank_to_host[inferred_rank]
            else:
                try:
                    cursor.execute("select value from TARGET_INFO_SYSTEM_ENV where name='Hostname' limit 1")
                    row = cursor.fetchone()
                    if row and row[0]:
                        host_name = row[0]
                except Exception:
                    host_name = None
            if host_name is None:
                host_name = os.path.splitext(file_name)[0]

            print(f'Host {file_count} Name: {host_name}')

            if host_name in HostName_To_GoalRank:
                goal_rank = HostName_To_GoalRank[host_name]
                GoalRank_To_NumOfGPUs[goal_rank] += 1
                if goal_rank not in nccl_events:
                    nccl_events[goal_rank] = {}
                    cupti_kernel_results[goal_rank] = {}
                    comm_init_events[goal_rank] = {}
                    events_counter[goal_rank] = {}
            else:
                goal_rank = len(HostName_To_GoalRank)
                HostName_To_GoalRank[host_name] = goal_rank
                GoalRank_To_NumOfGPUs[goal_rank] = 1
                nccl_events[goal_rank] = {}
                cupti_kernel_results[goal_rank] = {}
                comm_init_events[goal_rank] = {}
                events_counter[goal_rank] = {}

            
            pattern_nsys_profile_start = r"nsys profiling start, pid: (\d+)"
            pattern_nsys_profile_end = r"nsys profiling stopped, pid: (\d+)"

            pattern_Comm_Info = r'commHash (\S+) commId (\S+) rank (\d+) nranks (\d+) pid (\d+)'
            pattern_Comm_NumOfChannels = r'(\d+) coll channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer, pid (\d+)'

            pattern_Ring = r'commHash (\S+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)'
            pattern_Tree = r'commHash (\S+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)'

            # Find communicator info
            cursor.execute("SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'commHash%'")
            nvtx_events_results = cursor.fetchall()

            for row in nvtx_events_results:
                if row[0]:
                    match_Comm_Info = re.search(pattern_Comm_Info, row[0])
                    match_Comm_NumOfChannels = re.search(pattern_Comm_NumOfChannels, row[0])

                    match_Ring = re.search(pattern_Ring, row[0])
                    match_Tree = re.search(pattern_Tree, row[0])

                    if match_Comm_Info:  # 'commHash (\S+) commId (\S+) rank (\d+) nranks (\d+) pid (\d+)'
                        commHash = match_Comm_Info.group(1)
                        commId = match_Comm_Info.group(2)
                        my_rank = match_Comm_Info.group(3)
                        nranks = match_Comm_Info.group(4)
                        pid = match_Comm_Info.group(5)

                        ts_init_start = row[1] ## ns
                        ts_init_end = row[2] ## ns

                        if commId not in comm_info:
                            comm_info[commId] = {}
                            comm_info[commId]['nranks'] = int(nranks)
                            comm_info[commId]['gpuId_To_rank'] = {}
                            comm_info[commId]['rank_To_rankInfo'] = {}
                            comm_info[commId]['comm_index'] = len(comm_info) - 1

                        if pid not in pid_to_gpuId:
                            known_gpus += 1
                            gpuId = known_gpus
                            pid_to_gpuId[pid] = gpuId
                            commHash_to_commId[gpuId] = {}
                            stream_to_streamId[gpuId] = {}
                            Parse_State[gpuId] = 0  ## awaiting P2P or Group operations
                            nccl_events[goal_rank][gpuId] = {}    
                            cupti_kernel_results[goal_rank][gpuId] = {}
                            events_counter[goal_rank][gpuId] = {}

                        gpuId = pid_to_gpuId[pid]
                        comm_info[commId]['gpuId_To_rank'][gpuId] = my_rank
                        comm_info[commId]['rank_To_rankInfo'][my_rank] = {
                            'gpuId': gpuId,
                            'goal_rank': goal_rank,
                            'host_name': host_name,
                            'channel_info': {
                                'Ring': [],
                                'Tree': []
                            }
                        }

                        commHash_to_commId[gpuId][commHash] = commId
                        last_commId = commId

                        if gpuId not in comm_init_events[goal_rank]:
                            comm_init_events[goal_rank][gpuId] = {}
                            comm_init_events[goal_rank][gpuId]['ts_init_start'] = ts_init_start
                            comm_init_events[goal_rank][gpuId]['ts_init_end'] = ts_init_end

                    elif match_Comm_NumOfChannels:
                        num_P2P_channels_per_peer = match_Comm_NumOfChannels.group(4)
                        comm_info[last_commId]['NumOfP2PChannelsPerPeer'] = num_P2P_channels_per_peer

                    elif match_Ring:  ## 'commHash (\S+) Rings \[(\d+)\] (\d+)->(\d+)->(\d+) pid (\d+)'
                        commHash = match_Ring.group(1)
                        channel_Id = match_Ring.group(2)
                        previous_rank = match_Ring.group(3)
                        my_rank = match_Ring.group(4)
                        next_rank = match_Ring.group(5)
                        pid = match_Ring.group(6)

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        comm_info[commId]['rank_To_rankInfo'][my_rank]['channel_info']['Ring'].append(
                            {
                                'previous_rank': previous_rank,
                                'my_rank': my_rank,
                                'next_rank': next_rank,
                                'channel_Id': channel_Id
                            }
                        )

                    elif match_Tree:  ## 'commHash (\S+) Trees \[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(-?\d+)->(-?\d+) pid (\d+)'
                        commHash = match_Tree.group(1)
                        channel_Id = match_Tree.group(2)
                        child_1_rank = match_Tree.group(3)
                        child_2_rank = match_Tree.group(4)
                        child_3_rank = match_Tree.group(5)
                        my_rank = match_Tree.group(6)
                        parent_rank = match_Tree.group(7)
                        pid = match_Tree.group(8)

                        gpuId = pid_to_gpuId[pid]
                        commId = commHash_to_commId[gpuId][commHash]
                        comm_info[commId]['rank_To_rankInfo'][my_rank]['channel_info']['Tree'].append(
                            {
                                'child_1_rank': child_1_rank,
                                'child_2_rank': child_2_rank,
                                'child_3_rank': child_3_rank,
                                'my_rank': my_rank,
                                'parent_rank': parent_rank,
                                'channel_Id': channel_Id
                            }
                        )

            # Fallback for traces which do not include comm init NVTX markers (e.g., profiling
            # started after communicator creation): initialize pid->gpuId mapping using only
            # NCCL-relevant NVTX messages, otherwise we'd assign thousands of unrelated PIDs and
            # break the nccl-vs-kernel consistency checks below.
            if len(pid_to_gpuId) == 0:
                cursor.execute(
                    """
                    SELECT text FROM NVTX_EVENTS
                    WHERE text LIKE 'nccl%pid %'
                       OR text LIKE 'collType % pid %'
                       OR text LIKE 'nWarps % pid %'
                       OR text LIKE 'Bytes % pid %'
                       OR text LIKE 'nsys profiling%pid:%'
                    """
                )
                pid_re = re.compile(r"pid[: ]+(\\d+)")
                pids = set()
                for (txt,) in cursor.fetchall():
                    if not txt:
                        continue
                    m = pid_re.search(txt)
                    if m:
                        pids.add(m.group(1))

                for pid in sorted(pids, key=int):
                    if pid in pid_to_gpuId:
                        continue
                    known_gpus += 1
                    gpuId = known_gpus
                    pid_to_gpuId[pid] = gpuId
                    commHash_to_commId[gpuId] = {}
                    stream_to_streamId[gpuId] = {}
                    Parse_State[gpuId] = 0  ## awaiting P2P or Group operations
                    nccl_events[goal_rank][gpuId] = {}
                    cupti_kernel_results[goal_rank][gpuId] = {}
                    events_counter[goal_rank][gpuId] = {}
            
            # Fetch information about profiling interval
            cursor.execute("SELECT text, start FROM NVTX_EVENTS WHERE text LIKE 'nsys profiling%'")
            nvtx_events_results = cursor.fetchall()

            for row in nvtx_events_results:
                if row[0]:
                    match_profile_start = re.search(pattern_nsys_profile_start, row[0])
                    match_profile_end = re.search(pattern_nsys_profile_end, row[0])

                    if match_profile_start:
                        pid = match_profile_start.group(1)
                        if pid not in pid_to_gpuId:
                            continue
                        gpuId = pid_to_gpuId[pid]
                        ts_start = row[1] ## ns
                        assert gpuId not in profile_interval, f'[ERROR] gpuId {gpuId} already in profile_interval'
                        
                        profile_interval[gpuId] = {}
                        profile_interval[gpuId]["start_raw"] = ts_start
                        profile_interval[gpuId]["start"] = ts_start - profile_start_slack_ns
                    
                    elif match_profile_end:
                        pid = match_profile_end.group(1)
                        if pid not in pid_to_gpuId:
                            continue
                        gpuId = pid_to_gpuId[pid]
                        ts_end = row[1]
                        assert gpuId in profile_interval, f'[ERROR] gpuId {gpuId} not in profile_interval'
                        profile_interval[gpuId]["end_raw"] = ts_end
                        profile_interval[gpuId]["end"] = ts_end + profile_end_slack_ns

            # Fallback: for unpatched Megatron runs (or any trace set that doesn't include the
            # "nsys profiling start/stop" NVTX markers), Nsight Systems may still record the
            # effective capture window as DIAGNOSTIC_EVENT messages when capture-range is used.
            #
            # Using this window is important to keep the computed profiling interval consistent
            # across ranks/nodes; otherwise we fall back to min/max kernel timestamps which can
            # drift and cause large LGS mismatches.
            # Only apply this fallback for GPUs that don't already have an interval from NVTX.
            missing_profile_gpus = set(pid_to_gpuId.values()) - set(profile_interval.keys())
            if len(missing_profile_gpus) > 0:
                try:
                    cursor.execute(
                        """
                        SELECT timestamp, text
                        FROM DIAGNOSTIC_EVENT
                        WHERE text LIKE 'Profiling has started%'
                           OR text LIKE 'Profiling has stopped%'
                        ORDER BY timestamp
                        """
                    )
                    diag_rows = cursor.fetchall()
                except Exception:
                    diag_rows = []

                diag_start = None
                diag_end = None
                for ts, txt in diag_rows:
                    if not txt:
                        continue
                    if txt.startswith("Profiling has started"):
                        diag_start = ts if diag_start is None else min(diag_start, ts)
                    elif txt.startswith("Profiling has stopped"):
                        diag_end = ts if diag_end is None else max(diag_end, ts)

                if diag_start is not None and diag_end is not None and diag_end >= diag_start:
                    for gpuId in missing_profile_gpus:
                        profile_interval[gpuId] = {
                            "start_raw": int(diag_start),
                            "start": int(diag_start) - profile_start_slack_ns,
                            "end_raw": int(diag_end),
                            "end": int(diag_end) + profile_end_slack_ns,
                        }

            # If requested, override per-GPU profile windows with the global intersection window.
            if global_profile_window is not None and len(profile_interval) > 0:
                gs, ge = global_profile_window
                for gid, iv in profile_interval.items():
                    iv["start"] = int(gs)
                    iv["end"] = int(ge)
                    iv.setdefault("start_raw", int(gs))
                    iv.setdefault("end_raw", int(ge))

            cursor.execute('SELECT text, start, end FROM NVTX_EVENTS ORDER BY start')  ## row[0]: text, row[1]: ts_start, row[2]: ts_end
            nvtx_events_results = cursor.fetchall()

            pattern_nccl_AllReduce = r'ncclAllReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'
            pattern_nccl_Broadcast = r'ncclBroadcast\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), root (\d+), pid (\d+)'
            pattern_nccl_Reduce = r'ncclReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), root (\d+), pid (\d+)'
            pattern_nccl_AllGather = r'ncclAllGather\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), pid (\d+)'
            pattern_nccl_ReduceScatter = r'ncclReduceScatter\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'

            pattern_nccl_Send = r'ncclSend\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), receiver_rank (\d+), pid (\d+)'
            pattern_nccl_Recv = r'ncclRecv\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), sender_rank (\d+), pid (\d+)'

            pattern_nccl_GroupStart = r'ncclGroupStart\(\): pid (\d+)'
            pattern_nccl_GroupEnd = r'ncclGroupEnd\(\): pid (\d+)'

            pattern_Coll_Info = r'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
            pattern_Coll_Elem = r'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)'

            pattern_P2P_Elem = r'Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)'

            pattern_ncclKernel = r'ncclLaunchKernel\(\): pid (\d+)'

            pid_pattern = r'pid (\d+)'

            fallback_nranks = inferred_nranks if inferred_nranks is not None else int(os.environ.get("ATLAHS_FALLBACK_NRANKS", "1").strip() or "1")
            fallback_nranks = max(2, int(fallback_nranks))

            def ensure_synthetic_topology(ci, nchannels: int):
                """Ensure ci['rank_To_rankInfo'] exists for all ranks and has Ring/Tree channel_info."""
                nr = int(ci.get("nranks", 1))
                nchannels = max(1, int(nchannels))

                ri_map = ci.setdefault("rank_To_rankInfo", {})
                for r in range(nr):
                    rstr = str(r)
                    if rstr not in ri_map:
                        hn = inferred_rank_to_host.get(r, "unknown")
                        gr = HostName_To_GoalRank.get(hn, 0)
                        ri_map[rstr] = {
                            "gpuId": r,
                            "host_name": hn,
                            "goal_rank": gr,
                            "channel_info": {"Ring": [], "Tree": []},
                        }

                for r in range(nr):
                    rstr = str(r)
                    ri = ri_map[rstr]
                    chinfo = ri.setdefault("channel_info", {})
                    ring = chinfo.setdefault("Ring", [])
                    tree = chinfo.setdefault("Tree", [])

                    while len(ring) < nchannels:
                        ch = len(ring)
                        ring.append(
                            {
                                "previous_rank": str((r - 1) % nr),
                                "my_rank": rstr,
                                "next_rank": str((r + 1) % nr),
                                "channel_Id": str(ch),
                            }
                        )

                    while len(tree) < nchannels:
                        ch = len(tree)
                        parent = str((r - 1) // 2) if r > 0 else "-1"
                        c1 = str(2 * r + 1) if (2 * r + 1) < nr else "-1"
                        c2 = str(2 * r + 2) if (2 * r + 2) < nr else "-1"
                        tree.append(
                            {
                                "child_1_rank": c1,
                                "child_2_rank": c2,
                                "child_3_rank": "-1",
                                "my_rank": rstr,
                                "parent_rank": parent,
                                "channel_Id": str(ch),
                            }
                        )

            def ensure_pid(pid: str) -> int:
                """Assign a stable gpuId for this trace-local pid (only for NCCL-relevant pids)."""
                nonlocal known_gpus
                if pid not in pid_to_gpuId:
                    known_gpus += 1
                    gid = known_gpus
                    pid_to_gpuId[pid] = gid
                    commHash_to_commId[gid] = {}
                    stream_to_streamId[gid] = {}
                    Parse_State[gid] = 0  ## awaiting P2P or Group operations
                    nccl_events[goal_rank][gid] = {}
                    cupti_kernel_results[goal_rank][gid] = {}
                    events_counter[goal_rank][gid] = {}
                gid = pid_to_gpuId[pid]
                if inferred_rank is not None and inferred_rank not in global_rank_to_gpuId:
                    global_rank_to_gpuId[inferred_rank] = gid
                # Also record host-local pid mapping for log-based recovery.
                if host_name is not None:
                    try:
                        hostpid_to_gpuId[(str(host_name), int(pid))] = gid
                        gpuId_to_hostpid[int(gid)] = (str(host_name), int(pid))
                    except Exception:
                        pass
                return gid

            def ensure_comm_mapping(gpuId: int, commHash: str) -> Tuple[str, str]:
                """Return (commId, my_rank) and create minimal comm_info if init markers are missing."""
                if gpuId not in commHash_to_commId:
                    commHash_to_commId[gpuId] = {}
                commId = commHash_to_commId[gpuId].get(commHash)
                if commId is None:
                    commId = commHash
                    commHash_to_commId[gpuId][commHash] = commId

                if commId not in comm_info:
                    comm_info[commId] = {
                        "nranks": int(fallback_nranks),
                        "gpuId_To_rank": {},
                        "rank_To_rankInfo": {},
                        "comm_index": len(comm_info),
                        "_inferred": True,
                        "_observed_global_ranks": set(),
                        "_max_channels": 1,
                    }

                ci = comm_info[commId]
                try:
                    if gpuId in gpuId_to_hostpid:
                        commId_to_hostpids[str(commId)].add(gpuId_to_hostpid[int(gpuId)])
                except Exception:
                    pass
                r = inferred_rank if inferred_rank is not None else 0
                try:
                    ci["_observed_global_ranks"].add(int(r))
                except Exception:
                    pass
                # During parsing we don't know the communicator-local rank indexing for inferred comms
                # (no "commId rank nranks" marker). Use global rank as placeholder; we'll
                # finalize local ranks + ring topology after parsing all files.
                if gpuId not in ci["gpuId_To_rank"]:
                    ci["gpuId_To_rank"][gpuId] = str(r)
                my_rank = ci["gpuId_To_rank"][gpuId]

                if my_rank not in ci["rank_To_rankInfo"]:
                    ci["rank_To_rankInfo"][my_rank] = {
                        "gpuId": gpuId,
                        "goal_rank": goal_rank,
                        "host_name": host_name,
                        "channel_info": {"Ring": [], "Tree": []},
                    }
                else:
                    ri = ci["rank_To_rankInfo"][my_rank]
                    ri["gpuId"] = gpuId
                    ri["host_name"] = host_name
                    ri["goal_rank"] = goal_rank
                return commId, my_rank

            for row in tqdm(nvtx_events_results):
                if row[0]:
                    # If profiling interval is found, ignore all events outside of it
                    match_pid = re.search(pid_pattern, row[0])
                    if match_pid:
                        pid = match_pid.group(1)
                        if pid in pid_to_gpuId:
                            gpuId = pid_to_gpuId[pid]
                            if (gpuId in profile_interval) and \
                                (row[1] < profile_interval[gpuId]["start"] or row[1] > profile_interval[gpuId]["end"]):
                                continue

                    match_profile_start = re.search(pattern_nsys_profile_start, row[0])
                    match_profile_end = re.search(pattern_nsys_profile_end, row[0])

                    match_Comm_Info = re.search(pattern_Comm_Info, row[0])
                    match_Comm_NumOfChannels = re.search(pattern_Comm_NumOfChannels, row[0])

                    match_Ring = re.search(pattern_Ring, row[0])
                    match_Tree = re.search(pattern_Tree, row[0])

                    match_nccl_AllReduce = re.search(pattern_nccl_AllReduce, row[0])
                    match_nccl_Broadcast = re.search(pattern_nccl_Broadcast, row[0])
                    match_nccl_Reduce = re.search(pattern_nccl_Reduce, row[0])
                    match_nccl_AllGather = re.search(pattern_nccl_AllGather, row[0])
                    match_nccl_ReduceScatter = re.search(pattern_nccl_ReduceScatter, row[0])

                    match_nccl_Send = re.search(pattern_nccl_Send, row[0])
                    match_nccl_Recv = re.search(pattern_nccl_Recv, row[0])

                    match_nccl_GroupStart = re.search(pattern_nccl_GroupStart, row[0])
                    match_nccl_GroupEnd = re.search(pattern_nccl_GroupEnd, row[0])

                    match_Coll_Info = re.search(pattern_Coll_Info, row[0])
                    match_Coll_Elem = re.search(pattern_Coll_Elem, row[0])    

                    match_P2P_Elem = re.search(pattern_P2P_Elem, row[0])

                    match_ncclLaunchKernel = re.search(pattern_ncclKernel, row[0])


                    if match_nccl_AllReduce:  ## 'ncclAllReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), pid (\d+)'
                        commHash = match_nccl_AllReduce.group(1)
                        stream = match_nccl_AllReduce.group(2)
                        data_size = int(match_nccl_AllReduce.group(3))
                        type_size = int(match_nccl_AllReduce.group(4))
                        red_op = match_nccl_AllReduce.group(5)
                        pid = match_nccl_AllReduce.group(6)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = ensure_pid(pid)
                        commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllReduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllReduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'AllReduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'red_op': red_op,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['AllReduce']
                                    }
                                )    
                                
                                events_counter[goal_rank][gpuId][commId]['AllReduce'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllReduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllReduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'AllReduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_Broadcast:  ## 'ncclBroadcast\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), root (\d+)'
                        commHash = match_nccl_Broadcast.group(1)
                        stream = match_nccl_Broadcast.group(2)
                        data_size = int(match_nccl_Broadcast.group(3))
                        type_size = int(match_nccl_Broadcast.group(4))
                        root_rank = match_nccl_Broadcast.group(5)
                        pid = match_nccl_Broadcast.group(6)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = ensure_pid(pid)
                        commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Broadcast' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Broadcast'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'Broadcast',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'root_rank': root_rank,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['Broadcast']
                                    }
                                ) 
                                
                                events_counter[goal_rank][gpuId][commId]['Broadcast'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Broadcast' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Broadcast'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'Broadcast',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_Reduce:  ## 'ncclReduce\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+), root (\d+), pid (\d+)'
                        commHash = match_nccl_Reduce.group(1)
                        stream = match_nccl_Reduce.group(2)
                        data_size = int(match_nccl_Reduce.group(3))
                        type_size = int(match_nccl_Reduce.group(4))
                        red_op = match_nccl_Reduce.group(5)
                        root_rank = match_nccl_Reduce.group(6)
                        pid = match_nccl_Reduce.group(7)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = ensure_pid(pid)
                        commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Reduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Reduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'Reduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'red_op': red_op,
                                        'root_rank': root_rank,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['Reduce']
                                    }
                                )    
                                
                                events_counter[goal_rank][gpuId][commId]['Reduce'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Reduce' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Reduce'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'Reduce',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_AllGather:  ## 'ncclAllGather\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), pid (\d+)'
                        commHash = match_nccl_AllGather.group(1)
                        stream = match_nccl_AllGather.group(2)
                        data_size = int(match_nccl_AllGather.group(3))
                        type_size = int(match_nccl_AllGather.group(4))
                        pid = match_nccl_AllGather.group(5)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns
                        gpuId = ensure_pid(pid)
                        commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllGather' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllGather'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'AllGather',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['AllGather']
                                    }
                                )
                                
                                events_counter[goal_rank][gpuId][commId]['AllGather'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'AllGather' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['AllGather'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'AllGather',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_nccl_ReduceScatter:  ## 'ncclReduceScatter\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), red_op (\d+)'
                        commHash = match_nccl_ReduceScatter.group(1)
                        stream = match_nccl_ReduceScatter.group(2)
                        data_size = int(match_nccl_ReduceScatter.group(3))
                        type_size = int(match_nccl_ReduceScatter.group(4))
                        red_op = match_nccl_ReduceScatter.group(5)
                        pid = match_nccl_ReduceScatter.group(6)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        gpuId = ensure_pid(pid)
                        commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'ReduceScatter' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['ReduceScatter'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'ReduceScatter',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'data_size': data_size,
                                        'type_size': type_size,
                                        'red_op': red_op,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'seq': events_counter[goal_rank][gpuId][commId]['ReduceScatter']
                                    }
                                )
                                
                                events_counter[goal_rank][gpuId][commId]['ReduceScatter'] += 1

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                        elif Parse_State[gpuId] == 5:
                            Parse_State[gpuId] = 5

                        elif Parse_State[gpuId] == 1:
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'ReduceScatter' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['ReduceScatter'] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupColl',
                                        'coll_type': 'ReduceScatter',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'coll_events': []
                                    }
                                ) 

                                last_Coll_streamId[gpuId] = streamId
                                last_update[gpuId] = 'Coll'

                                Parse_State[gpuId] = 5

                    elif match_Coll_Info: 
                        ## 'collType (\d+) root (\d+) redOp (\d+) algo (\d+) proto (\d+) commHash (\S+) stream (\S+) data_size (\d+) type_size (\d+) chunkSize (\d+) chunkCount (\d+) chunkSteps (\d+) sliceSteps (\d+) stepSize (\d+) pid (\d+)'
                        collType = int(match_Coll_Info.group(1))
                        root_rank = int(match_Coll_Info.group(2))
                        redOp = int(match_Coll_Info.group(3))
                        algo = match_Coll_Info.group(4)
                        proto = match_Coll_Info.group(5)
                        commHash = match_Coll_Info.group(6)
                        stream = match_Coll_Info.group(7)
                        data_size = int(match_Coll_Info.group(8))
                        type_size = int(match_Coll_Info.group(9))
                        
                        chunkSteps = int(match_Coll_Info.group(12))
                        sliceSteps = int(match_Coll_Info.group(13))
                        stepSize = int(match_Coll_Info.group(14)) 
                        pid = match_Coll_Info.group(15)

                        gpuId = ensure_pid(pid)
                        commId, _ = ensure_comm_mapping(gpuId, commHash)
                        
                        if Parse_State[gpuId] == 0:
                            if gpuId not in last_Coll_streamId:
                                continue
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['algorithm'] = algo
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['protocol'] = proto
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['chunkSteps'] = chunkSteps
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['sliceSteps'] = sliceSteps
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['stepSize'] = stepSize

                        elif Parse_State[gpuId] == 6:
                            if gpuId not in last_Coll_streamId:
                                continue
                            CollType = nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_type']

                            if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                            if CollType not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId][CollType] = 0

                            assert commHash_to_commId[gpuId][commHash] == nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['commId'], 'not the same comm in groupoperation'
                            # Skip stream assertions for nil streams (NCCL 2.28 internal markers sometimes have nil stream pointers)
                            if stream != '(nil)' and stream in stream_to_streamId.get(gpuId, {}):
                                assert stream_to_streamId[gpuId][stream] == nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['streamId'], 'not the same stream in group operation 1'
                                assert stream_to_streamId[gpuId][stream] == last_Coll_streamId[gpuId], 'not the same stream in group operation 2'

                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'].append(
                                {
                                    'algorithm': algo,
                                    'protocol': proto,
                                    'data_size': data_size,
                                    'type_size': type_size,
                                    'root_rank': root_rank,
                                    'red_op': redOp,
                                    'seq': events_counter[goal_rank][gpuId][commId][CollType],
                                    'chunkSteps': chunkSteps,
                                    'sliceSteps': sliceSteps,
                                    'stepSize': stepSize
                                }
                            )

                            events_counter[goal_rank][gpuId][commId][CollType] += 1

                    elif match_Coll_Elem: ## 'nWarps (\d+) count (\d+) chunkCount (\d+) workCount (\d+) lastChunkCount (\d+) workOffset (\d+) sendbuff (\d+) recvbuff (\d+) pid (\d+)'
                        nWarps = int(match_Coll_Elem.group(1))
                        count = int(match_Coll_Elem.group(2))
                        chunkCount = int(match_Coll_Elem.group(3))
                        workCount = int(match_Coll_Elem.group(4))
                        lastChunkCount = int(match_Coll_Elem.group(5))
                        workOffset = int(match_Coll_Elem.group(6))
                        sendbuff = int(match_Coll_Elem.group(7))
                        recvbuff = int(match_Coll_Elem.group(8))
                        pid = match_Coll_Elem.group(9)

                        gpuId = ensure_pid(pid)

                        if Parse_State[gpuId] == 0:
                            if gpuId not in last_Coll_streamId:
                                continue
                            if 'elems' not in nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]:
                                nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['elems'] = []

                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['elems'].append(
                                {
                                    'count': count,
                                    'chunkCount': chunkCount,
                                    'workCount': workCount,
                                    'lastChunkCount': lastChunkCount,
                                    'workOffset': workOffset,
                                    'sendbuff': sendbuff,
                                    'recvbuff': recvbuff,
                                }
                            )
                            ev = nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]
                            ci = comm_info.get(ev.get("commId"))
                            if ci is not None and ci.get("_inferred"):
                                ci["_max_channels"] = max(int(ci.get("_max_channels", 1)), len(ev.get("elems", [])))

                        elif Parse_State[gpuId] == 6:
                            if gpuId not in last_Coll_streamId:
                                continue
                            if 'elems' not in nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'][-1]:
                                nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'][-1]['elems'] = []
                            
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['coll_events'][-1]['elems'].append(
                                {
                                    'count': count,
                                    'chunkCount': chunkCount,
                                    'workCount': workCount,
                                    'lastChunkCount': lastChunkCount,
                                    'workOffset': workOffset,
                                    'sendbuff': sendbuff,
                                    'recvbuff': recvbuff,
                                }
                            )
                            group_ev = nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]
                            ce = group_ev['coll_events'][-1]
                            ci = comm_info.get(group_ev.get("commId"))
                            if ci is not None and ci.get("_inferred"):
                                ci["_max_channels"] = max(int(ci.get("_max_channels", 1)), len(ce.get("elems", [])))

                    elif match_nccl_GroupStart:
                        pid = match_nccl_GroupStart.group(1)

                        if pid not in pid_to_gpuId:
                            known_gpus += 1
                            gpuId = known_gpus
                            pid_to_gpuId[pid] = gpuId
                            commHash_to_commId[gpuId] = {}
                            stream_to_streamId[gpuId] = {}
                            Parse_State[gpuId] = 0  ## awaiting P2P or Group operations
                            nccl_events[goal_rank][gpuId] = {}    
                            cupti_kernel_results[goal_rank][gpuId] = {}
                            events_counter[goal_rank][gpuId] = {}

                        gpuId = pid_to_gpuId[pid]

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0

                        if Parse_State[gpuId] == 0:
                            ts_group_start[gpuId] = row[1] ## ns
                            Parse_State[gpuId] = 1  ## awaiting ncclColl or ncclP2P, ignore ncclGroupStart/ncclGroupEnd in between

                        elif Parse_State[gpuId] == 2:
                            Parse_State[gpuId] = 3

                        elif Parse_State[gpuId] == 7:
                            Parse_State[gpuId] = 8

                    elif match_nccl_GroupEnd:
                        pid = match_nccl_GroupEnd.group(1)
                        gpuId = ensure_pid(pid)

                        if Parse_State[gpuId] == 3:
                            Parse_State[gpuId] = 2

                        elif Parse_State[gpuId] == 2:
                            ts_group_end[gpuId] = row[2] ## ns
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['ts_end'] = ts_group_end[gpuId]
                            Parse_State[gpuId] = 4

                        elif Parse_State[gpuId] == 5:
                            ts_group_end[gpuId] = row[2]  ## ns
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['ts_end'] = ts_group_end[gpuId]
                            Parse_State[gpuId] = 6

                        elif Parse_State[gpuId] == 1:  ## in case directly call ncclGroupEnd() after ncclGroupStart() 
                            Parse_State[gpuId] = 0

                        elif Parse_State[gpuId] == 8:
                            Parse_State[gpuId] = 7

                    elif match_nccl_Send:  ## 'ncclSend\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), receiver_rank (\d+), pid (\d+)'
                        commHash = match_nccl_Send.group(1)
                        stream = match_nccl_Send.group(2)
                        data_size = int(match_nccl_Send.group(3))
                        type_size = int(match_nccl_Send.group(4))
                        peer_rank = match_nccl_Send.group(5)
                        pid = match_nccl_Send.group(6)

                        gpuId = ensure_pid(pid)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0
                        
                        if Parse_State[gpuId] == 1:  ## Group P2P
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Send'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                    events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 2

                                last_P2P_streamId[gpuId] = streamId    
                                last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 2:  ## Group P2P
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)
                            streamId = stream_to_streamId[gpuId][stream]

                            if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId]['Send'] = {}

                            if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                            Parse_State[gpuId] = 2

                            last_P2P_streamId[gpuId] = streamId    
                            last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 0:  ## Single P2P
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Send' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Send'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Send']:
                                    events_counter[goal_rank][gpuId][commId]['Send'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 7

                                last_P2P_streamId[gpuId] = streamId    
                                last_update[gpuId] = 'P2P'

                    elif match_nccl_Recv:  ## 'ncclRecv\(\): commHash (\S+), stream (\S+), data_size (\d+), type_size (\d+), sender_rank (\d+)'
                        commHash = match_nccl_Recv.group(1)
                        stream = match_nccl_Recv.group(2)
                        data_size = int(match_nccl_Recv.group(3))
                        type_size = int(match_nccl_Recv.group(4))
                        peer_rank = match_nccl_Recv.group(5)
                        pid = match_nccl_Recv.group(6)

                        gpuId = ensure_pid(pid)

                        ts_start = row[1] ## ns
                        ts_end = row[2] ## ns

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 6 or Parse_State[gpuId] == 9:
                            Parse_State[gpuId] = 0
                        
                        if Parse_State[gpuId] == 1:  ## Group P2P
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                    events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_group_start[gpuId],
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 2

                                last_P2P_streamId[gpuId] = streamId
                                last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 2:  ## Group P2P
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)
                            streamId = stream_to_streamId[gpuId][stream]

                            if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                            if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                            Parse_State[gpuId] = 2

                            last_P2P_streamId[gpuId] = streamId
                            last_update[gpuId] = 'P2P'

                        elif Parse_State[gpuId] == 0:  ## Single P2P
                            commId, my_rank = ensure_comm_mapping(gpuId, commHash)

                            if comm_info[commId]['nranks'] > 1 and data_size > 0:
                                if commId not in events_counter[goal_rank][gpuId]:
                                    events_counter[goal_rank][gpuId][commId] = {}

                                if 'Recv' not in events_counter[goal_rank][gpuId][commId]:
                                    events_counter[goal_rank][gpuId][commId]['Recv'] = {}

                                if peer_rank not in events_counter[goal_rank][gpuId][commId]['Recv']:
                                    events_counter[goal_rank][gpuId][commId]['Recv'][peer_rank] = 0

                                if stream not in stream_to_streamId[gpuId]:
                                    stream_to_streamId[gpuId][stream] = len(stream_to_streamId[gpuId])

                                streamId = stream_to_streamId[gpuId][stream]
                                if streamId not in nccl_events[goal_rank][gpuId]:
                                    nccl_events[goal_rank][gpuId][streamId] = []

                                nccl_events[goal_rank][gpuId][streamId].append(
                                    {
                                        'event_type': 'GroupP2P',
                                        'commId': commId,
                                        'comm_index': comm_info[commId]['comm_index'],
                                        'streamId': streamId,
                                        'my_rank': my_rank,
                                        'gpuId': gpuId,
                                        'ts_start': ts_start,
                                        'ts_end': ts_end,
                                        'P2P_events': []
                                    }
                                ) 
                                
                                Parse_State[gpuId] = 7

                                last_P2P_streamId[gpuId] = streamId    
                                last_update[gpuId] = 'P2P'

                    elif match_P2P_Elem:  ## 'Bytes (\d+) nWarps (\d+) p2pType (\d+) peer (\d+) proto (\d+) countHi32 (\d+) countLo32 (\d+) chunkSize (\d+) pid (\d+)'
                        p2pType = match_P2P_Elem.group(3)
                        peer_rank = match_P2P_Elem.group(4)
                        proto = match_P2P_Elem.group(5)
                        countHi32 = int(match_P2P_Elem.group(6))
                        countLo32 = int(match_P2P_Elem.group(7))
                        chunkSize = int(match_P2P_Elem.group(8))
                        pid  = match_P2P_Elem.group(9)

                        gpuId = ensure_pid(pid)
                        if gpuId not in last_P2P_streamId:
                            continue
                        commId = nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['commId']

                        if p2pType == '1':
                            p2p_type = 'Send' 
                        elif p2pType == '2':
                            p2p_type = 'Recv' 

                        if Parse_State[gpuId] == 4 or Parse_State[gpuId] == 7 or Parse_State[gpuId] == 9:
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['P2P_events'].append(
                                {
                                    'p2p_type': p2p_type,
                                    'peer_rank': peer_rank,
                                    'protocol': proto,
                                    'countHi32': countHi32,
                                    'countLo32': countLo32,
                                    'chunkSize': chunkSize,
                                    'count': countHi32 * 2**32 + countLo32,
                                    'seq': events_counter[goal_rank][gpuId][commId][p2p_type][peer_rank]
                                }
                            )

                            if Parse_State[gpuId] == 7:
                                Parse_State[gpuId] = 9

                            events_counter[goal_rank][gpuId][commId][p2p_type][peer_rank] += 1

                    elif match_ncclLaunchKernel:
                        pid = match_ncclLaunchKernel.group(1)

                        gpuId = ensure_pid(pid)

                        ts_kernel = row[2] ## ns

                        if gpuId not in last_update:
                            continue

                        if last_update[gpuId] == 'Coll':
                            if gpuId not in last_Coll_streamId:
                                continue
                            nccl_events[goal_rank][gpuId][last_Coll_streamId[gpuId]][-1]['ts_kernel'] = ts_kernel

                        elif last_update[gpuId] == 'P2P':
                            if gpuId not in last_P2P_streamId:
                                continue
                            nccl_events[goal_rank][gpuId][last_P2P_streamId[gpuId]][-1]['ts_kernel'] = ts_kernel
                
                    
            
            cursor.execute('SELECT globalPid, pid FROM PROCESSES')
            globalPid_pids = cursor.fetchall()
            pid_dict = {row[0]: row[1] for row in globalPid_pids}
            
            cursor.execute('SELECT id, value FROM StringIds')
            string_ids = cursor.fetchall()
            string_dict = {row[0]: row[1] for row in string_ids}
            
            cursor.execute('SELECT start, end, streamId, globalPid, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL')
            cupti_kernel_events = cursor.fetchall()
            for row in cupti_kernel_events:
                start, end, streamId, globalPid, demangled_name = row
                if string_dict[demangled_name].startswith('ncclKernel') or string_dict[demangled_name].startswith('ncclDevKernel'):
                    fields = string_dict[demangled_name].replace('(', '_').replace(')', '_').split('_')
                    pid = pid_dict[globalPid]
                    gpuId = ensure_pid(str(pid))

                    # Do not include the kernel if it is not inside the profile interval
                    if gpuId in profile_interval and \
                        (end < profile_interval[gpuId]['start'] or start > profile_interval[gpuId]['end']):
                        continue

                    if streamId not in cupti_kernel_results[goal_rank][gpuId]:
                        cupti_kernel_results[goal_rank][gpuId][streamId] = [] 

                    cupti_kernel_results[goal_rank][gpuId][streamId].append({
                        'gpu_event_type': fields[1],
                        'ts_gpu_start': start, ## ns
                        'ts_gpu_end': end, ## ns
                    })

            conn.close()

            for gpuId in nccl_events[goal_rank].keys():
                len_nccl_events = sum([len(v) for v in nccl_events[goal_rank][gpuId].values()])
                len_cupti_kernel_results = sum([len(v) for v in cupti_kernel_results[goal_rank][gpuId].values()])
                if len_nccl_events != len_cupti_kernel_results and len_nccl_events > 0 and len_cupti_kernel_results > 0:
                    # Late-capture traces can contain CUDA kernels which don't have matching NCCL NVTX
                    # metadata (e.g., kernels already in-flight when capture starts). In that case,
                    # a naive "aggregate all kernels in [ts_kernel_i, ts_kernel_{i+1})" can overestimate
                    # NCCL time by accidentally absorbing unrelated kernels and/or very long windows.
                    #
                    # Instead, derive one synthetic GPU interval per NCCL event using:
                    # - kernels on the same CUDA stream as the NCCL event
                    # - kernel type matching (gpu_event_type == coll_type where possible)
                    # - host-side NVTX duration as a fallback to repair missing kernel end timestamps
                    msg = f'Host {goal_rank} gpu: {gpuId} Different number of events in nccl and cupti kernel results {len_nccl_events} != {len_cupti_kernel_results}'
                    if os.environ.get("ATLAHS_SYNTHETIC_CUPTI_ON_MISMATCH", "0").strip() != "1":
                        # Default: keep the raw CUPTI kernels. Higher-level processing may still be
                        # able to merge/align streams after filtering out non-NCCL streams.
                        print(f'[WARN] {msg} (keeping raw CUPTI kernels; set ATLAHS_SYNTHETIC_CUPTI_ON_MISMATCH=1 to synthesize)')
                        continue

                    print(f'[WARN] {msg} (synthesizing CUPTI per NCCL event)')

                    # Flatten NCCL events keeping stable per-stream indices.
                    nccl_refs = []
                    for sid, evs in nccl_events[goal_rank][gpuId].items():
                        for idx, ev in enumerate(evs):
                            t = ev.get("ts_kernel", None)
                            if t is None:
                                t = ev.get("ts_start", 0)
                            nccl_refs.append((int(t), int(sid), int(idx)))
                    nccl_refs.sort(key=lambda x: x[0])

                    # Index CUPTI kernels by stream and keep them time-sorted.
                    kernels_by_stream = {}
                    for sid_str, kev in cupti_kernel_results[goal_rank][gpuId].items():
                        sid_int = int(sid_str)
                        kernels = []
                        for k in kev:
                            kernels.append(
                                {
                                    "gpu_event_type": k.get("gpu_event_type", None),
                                    "ts_gpu_start": int(k["ts_gpu_start"]),
                                    "ts_gpu_end": int(k["ts_gpu_end"]),
                                }
                            )
                        kernels.sort(key=lambda x: x["ts_gpu_start"])
                        kernels_by_stream[sid_int] = kernels

                    # Assign per-NCCL-event GPU intervals, avoiding cross-stream contamination.
                    assigned = {}
                    # Pre-compute per-event expected type + a sane duration fallback.
                    expected = {}
                    for sid_str, evs in nccl_events[goal_rank][gpuId].items():
                        sid_int = int(sid_str)
                        for idx, ev in enumerate(evs):
                            if ev.get("event_type") == "GroupColl":
                                etype = ev.get("coll_type")
                            elif ev.get("event_type") == "GroupP2P":
                                etype = "SendRecv"
                            else:
                                etype = ev.get("event_type")
                            t0 = int(ev.get("ts_kernel", ev.get("ts_start", 0)))
                            # Host-side duration is imperfect but better than zero-length kernels.
                            host_dur = int(ev.get("ts_end", t0)) - int(ev.get("ts_kernel", t0))
                            if host_dur <= 0:
                                host_dur = int(ev.get("ts_end", t0)) - int(ev.get("ts_start", t0))
                            if host_dur <= 0:
                                host_dur = 1
                            expected[(sid_int, idx)] = (etype, t0, host_dur)

                    for i, (t0, sid, idx) in enumerate(nccl_refs):
                        t1 = nccl_refs[i + 1][0] if (i + 1) < len(nccl_refs) else 2**63 - 1
                        etype, t0_exp, host_dur = expected.get((sid, idx), (None, t0, 1))

                        # Candidate kernels: same stream, starting within [t0, t1).
                        stream_kernels = kernels_by_stream.get(sid, [])
                        cand_ixs = [j for j, k in enumerate(stream_kernels) if (t0 <= k["ts_gpu_start"] < t1)]

                        # Prefer kernels whose gpu_event_type matches NCCL event type, but keep a fallback.
                        typed_ixs = []
                        if etype is not None:
                            typed_ixs = [j for j in cand_ixs if stream_kernels[j].get("gpu_event_type") == etype]
                        use_ixs = typed_ixs if typed_ixs else cand_ixs

                        if use_ixs:
                            gpu_start = min(stream_kernels[j]["ts_gpu_start"] for j in use_ixs)

                            # Repair missing kernel end timestamps using the next kernel start on the same stream.
                            gpu_end = gpu_start
                            for j in use_ixs:
                                ks = stream_kernels[j]["ts_gpu_start"]
                                ke = stream_kernels[j]["ts_gpu_end"]
                                if ke <= ks:
                                    nxt = stream_kernels[j + 1]["ts_gpu_start"] if (j + 1) < len(stream_kernels) else t1
                                    ke = min(int(t1), int(nxt))
                                    if ke <= ks:
                                        ke = ks + 1
                                gpu_end = max(gpu_end, ke)
                        else:
                            # No matching kernel recorded: fall back to a minimal interval at ts_kernel.
                            gpu_start = int(t0_exp)
                            gpu_end = gpu_start + max(1, host_dur)

                        assigned[(sid, idx)] = (gpu_start, gpu_end)

                    # Build per-NCCL-stream synthetic CUPTI lists that match length+types.
                    new_cupti = {}
                    for sid, evs in nccl_events[goal_rank][gpuId].items():
                        sid_int = int(sid)
                        new_cupti[sid_int] = []
                        for idx, ev in enumerate(evs):
                            if ev.get("event_type") == "GroupColl":
                                etype = ev.get("coll_type")
                            elif ev.get("event_type") == "GroupP2P":
                                etype = "SendRecv"
                            else:
                                etype = ev.get("event_type")
                            gs, ge = assigned.get((sid_int, int(idx)), (ev.get("ts_start", 0), ev.get("ts_start", 0)))
                            new_cupti[sid_int].append({"gpu_event_type": etype, "ts_gpu_start": int(gs), "ts_gpu_end": int(ge)})

                    cupti_kernel_results[goal_rank][gpuId] = new_cupti

    # Finalize inferred communicator metadata (after all files are parsed):
    # - determine comm membership from observed global ranks
    # - build communicator-local ranks and synthetic ring/tree topology
    for commId, ci in comm_info.items():
        if not ci.get("_inferred"):
            continue
        members = sorted(int(r) for r in ci.get("_observed_global_ranks", set()))
        if len(members) == 0:
            continue
        members = [r for r in members if r in global_rank_to_gpuId]
        if len(members) == 0:
            continue

        local_of_global = {gr: i for i, gr in enumerate(members)}
        nr = len(members)
        ci["nranks"] = nr

        nch = max(1, int(ci.get("_max_channels", 1)))
        new_rank_info = {}
        new_gpu_to_rank = {}

        for gr in members:
            lr = local_of_global[gr]
            gid = global_rank_to_gpuId[gr]
            hn = inferred_rank_to_host.get(gr)
            if not hn:
                node = inferred_rank_to_node.get(gr, 0)
                hn = f"node{node}"
            grank = HostName_To_GoalRank.get(hn, 0)
            new_gpu_to_rank[int(gid)] = str(lr)

            ring = []
            tree = []
            for ch in range(nch):
                ring.append(
                    {
                        "previous_rank": str((lr - 1) % nr),
                        "my_rank": str(lr),
                        "next_rank": str((lr + 1) % nr),
                        "channel_Id": str(ch),
                    }
                )
                parent = str((lr - 1) // 2) if lr > 0 else "-1"
                c1 = str(2 * lr + 1) if (2 * lr + 1) < nr else "-1"
                c2 = str(2 * lr + 2) if (2 * lr + 2) < nr else "-1"
                tree.append(
                    {
                        "child_1_rank": c1,
                        "child_2_rank": c2,
                        "child_3_rank": "-1",
                        "my_rank": str(lr),
                        "parent_rank": parent,
                        "channel_Id": str(ch),
                    }
                )

            new_rank_info[str(lr)] = {
                "gpuId": int(gid),
                "goal_rank": int(grank),
                "host_name": hn,
                "channel_info": {"Ring": ring, "Tree": tree},
            }

        ci["rank_To_rankInfo"] = new_rank_info
        ci["gpuId_To_rank"] = new_gpu_to_rank

        # Drop non-serializable / internal-only fields after finalization.
        ci.pop("_observed_global_ranks", None)
        ci.pop("_max_channels", None)
        ci.pop("_inferred", None)

    # Optional: reconstruct/init topology from NCCL debug logs when INIT NVTX markers are missing.
    # This is useful when nsys started tracing after communicator creation (capture-range),
    # so the trace contains API/ENQUEUE markers but not Rings/Trees/CommInit markers.
    if use_nccl_debug_topology and nccl_debug_log_glob:
        topo_by_commhash = _parse_nccl_debug_topology_from_logs(nccl_debug_log_glob)
        if len(topo_by_commhash) > 0:
            topo_id_to_hostpids = {}
            for tid, topo in topo_by_commhash.items():
                try:
                    topo_id_to_hostpids[tid] = set((topo.get("pid_to_rank") or {}).keys())
                except Exception:
                    topo_id_to_hostpids[tid] = set()

            applied = 0
            for commId, ci in comm_info.items():
                topo_key = commId
                topo = topo_by_commhash.get(topo_key)
                if topo is None:
                    # Heuristic: match commId (NVTX commHash) to a log-commId by the set of (host,pid)
                    # that participates in that communicator.
                    try:
                        comm_set = commId_to_hostpids.get(str(commId), set())
                    except Exception:
                        comm_set = set()
                    if comm_set:
                        best = None
                        best_n = 0
                        for tid, hp in topo_id_to_hostpids.items():
                            n = len(comm_set & hp)
                            if n > best_n:
                                best_n = n
                                best = tid
                        if best is not None and best_n == len(comm_set) and len(comm_set) >= 2:
                            topo_key = best
                            topo = topo_by_commhash.get(best)
                if topo is None:
                    continue

                ring_map = topo.get("ring", {})
                tree_map = topo.get("tree", {})
                pid_to_rank = topo.get("pid_to_rank", {})
                if len(ring_map) == 0 and len(tree_map) == 0:
                    continue

                # Update nranks if present in logs.
                if topo.get("nranks") is not None:
                    try:
                        ci["nranks"] = int(topo["nranks"])
                    except Exception:
                        pass

                # Build per-rank channel lists, sorted by channel id.
                by_rank_ring: Dict[str, List[Dict[str, str]]] = defaultdict(list)
                for (r, ch), v in ring_map.items():
                    by_rank_ring[str(r)].append(dict(v))
                for r in by_rank_ring:
                    by_rank_ring[r].sort(key=lambda d: int(d.get("channel_Id", "0")))

                by_rank_tree: Dict[str, List[Dict[str, str]]] = defaultdict(list)
                for (r, ch), v in tree_map.items():
                    by_rank_tree[str(r)].append(dict(v))
                for r in by_rank_tree:
                    by_rank_tree[r].sort(key=lambda d: int(d.get("channel_Id", "0")))

                # Best-effort: use NCCL debug logs to map host-local pid -> NCCL rank,
                # and then connect that pid to the trace gpuId seen in SQLite.
                rank_to_host: Dict[str, str] = {}
                rank_to_gpuId: Dict[str, int] = {}
                if isinstance(pid_to_rank, dict) and len(pid_to_rank) > 0:
                    for (h, p), rint in pid_to_rank.items():
                        rstr = str(int(rint))
                        try:
                            rank_to_host.setdefault(rstr, str(h))
                        except Exception:
                            pass
                        try:
                            gid = hostpid_to_gpuId.get((str(h), int(p)))
                            if gid is not None:
                                rank_to_gpuId[rstr] = int(gid)
                        except Exception:
                            pass

                # Prefer ranks from pid mapping; otherwise fall back to ranks present in ring/tree.
                ranks_seen = set(list(by_rank_ring.keys()) + list(by_rank_tree.keys()) + list(rank_to_host.keys()) + list(rank_to_gpuId.keys()))
                if len(ranks_seen) == 0:
                    continue

                new_rank_info: Dict[str, Dict[str, object]] = {}
                new_gpu_to_rank: Dict[int, str] = {}
                for r in sorted(ranks_seen, key=lambda x: int(x)):
                    host = rank_to_host.get(r, "unknown")
                    grank = HostName_To_GoalRank.get(host, 0) if host != "unknown" else 0
                    gid = int(rank_to_gpuId.get(r, -1))
                    if gid >= 0:
                        new_gpu_to_rank[gid] = r
                    new_rank_info[r] = {
                        "gpuId": gid,
                        "goal_rank": int(grank),
                        "host_name": host,
                        "channel_info": {"Ring": by_rank_ring.get(r, []), "Tree": by_rank_tree.get(r, [])},
                    }

                ci["rank_To_rankInfo"] = new_rank_info
                ci["gpuId_To_rank"] = new_gpu_to_rank

                applied += 1

            if applied > 0:
                print(f"[INFO] Applied NCCL debug-log topology for {applied} communicator(s) (glob: {nccl_debug_log_glob})")

    # Ensure comm_info is JSON-serializable (intermediate outputs are JSON).
    for _, ci in comm_info.items():
        # Internal inference fields:
        ci.pop("_observed_global_ranks", None)
        ci.pop("_max_channels", None)

    # After log-based recovery, drop events that belong to nranks<=1 communicators.
    # These often appear in late-traced runs because fallback nranks is >1 during parsing
    # (before we apply the recovered topology), but the real communicator is local-only.
    for grank, gpus in list(nccl_events.items()):
        for gid, streams in list(gpus.items()):
            for sid, evs in list(streams.items()):
                new_evs = []
                for ev in evs:
                    cid = ev.get("commId")
                    try:
                        nr = int(comm_info.get(cid, {}).get("nranks", 2))
                    except Exception:
                        nr = 2
                    if nr <= 1:
                        continue
                    new_evs.append(ev)
                if len(new_evs) == 0:
                    del nccl_events[grank][gid][sid]
                else:
                    nccl_events[grank][gid][sid] = new_evs
            if len(nccl_events[grank][gid]) == 0:
                del nccl_events[grank][gid]
        if len(nccl_events[grank]) == 0:
            del nccl_events[grank]

    # Ensure every GPU has a comm-init interval. Some trace sets start profiling after
    # communicator creation, so "commHash ... commId ... rank ..." markers are missing.
    # Downstream dependency builders still require a (ts_init_start, ts_init_end) anchor.
    for goal_rank, gpus in nccl_events.items():
        if goal_rank not in comm_init_events:
            comm_init_events[goal_rank] = {}
        for gpuId in gpus.keys():
            if gpuId in comm_init_events[goal_rank]:
                continue
            # Prefer earliest GPU kernel start; fall back to earliest host event start.
            earliest = None
            for _, stream_events in cupti_kernel_results.get(goal_rank, {}).get(gpuId, {}).items():
                for e in stream_events:
                    t = e.get("ts_gpu_start")
                    if t is None:
                        continue
                    earliest = t if earliest is None else min(earliest, t)
            if earliest is None:
                for _, stream_events in nccl_events[goal_rank][gpuId].items():
                    for e in stream_events:
                        t = e.get("ts_start")
                        if t is None:
                            continue
                        earliest = t if earliest is None else min(earliest, t)
            if earliest is None:
                earliest = 0
            comm_init_events[goal_rank][gpuId] = {"ts_init_start": int(earliest), "ts_init_end": int(earliest)}

    # Ensure every GPU has a finite profile interval.
    #
    # Some trace sets do not contain the NVTX "nsys profiling start/stop" markers.
    # For those, fall back to the min/max NCCL kernel timestamps.
    #
    # Optionally, clamp the interval length via ATLAHS_PROFILE_WINDOW_NS.
    clamp_window_ns = os.environ.get("ATLAHS_PROFILE_WINDOW_NS", "").strip()
    try:
        clamp_window_ns = int(clamp_window_ns) if clamp_window_ns != "" else None
    except ValueError:
        clamp_window_ns = None

    for goal_rank, gpus in cupti_kernel_results.items():
        for gpuId, streams in gpus.items():
            mn = None
            mx = None
            for _, evs in streams.items():
                for e in evs:
                    s = e.get("ts_gpu_start")
                    t = e.get("ts_gpu_end")
                    if s is not None:
                        mn = s if mn is None else min(mn, s)
                    if t is not None:
                        mx = t if mx is None else max(mx, t)
            if mn is None or mx is None:
                continue

            if gpuId not in profile_interval:
                profile_interval[gpuId] = {"start": int(mn), "end": int(mx)}
            else:
                if "start" not in profile_interval[gpuId]:
                    profile_interval[gpuId]["start"] = int(mn)
                if "end" not in profile_interval[gpuId]:
                    profile_interval[gpuId]["end"] = int(mx)

            if clamp_window_ns is not None and clamp_window_ns > 0:
                s = int(profile_interval[gpuId]["start"])
                profile_interval[gpuId]["end"] = min(int(profile_interval[gpuId]["end"]), s + int(clamp_window_ns))

    # If we pre-scanned a global intersection window, enforce it now (after any kernel-based
    # fallback interval construction). This intentionally does NOT trim events; it only ensures
    # downstream computations (e.g., exposed comm time) use a consistent window across nodes.
    if global_profile_window is not None and len(profile_interval) > 0:
        gs, ge = global_profile_window
        for _, iv in profile_interval.items():
            iv["start"] = int(gs)
            iv["end"] = int(ge)
            iv.setdefault("start_raw", int(gs))
            iv.setdefault("end_raw", int(ge))

    # Optional: deterministically remap goal ranks + GPU IDs using the largest communicator
    # (max nranks) as the stable reference.
    #
    # This helps when file/PID discovery order differs across trace sets and would otherwise
    # lead to different host->goal_rank and pid->gpuId assignments.
    det_map_enabled = os.environ.get("ATLAHS_DETERMINISTIC_RANK_GPU_MAP", "0").strip() == "1"
    if det_map_enabled and len(comm_info) > 0:
        ref_commId = max(comm_info.keys(), key=lambda cid: int(comm_info[cid].get("nranks", 0)))
        ref_ci = comm_info[ref_commId]
        ref_rank_info = ref_ci.get("rank_To_rankInfo", {})
        if len(ref_rank_info) > 0:
            # Determine stable host ordering from ref comm: sort hosts by their minimum rank.
            host_to_ranks = defaultdict(list)
            for rank_str, ri in ref_rank_info.items():
                host_to_ranks[ri["host_name"]].append(int(rank_str))
            host_order = sorted(host_to_ranks.keys(), key=lambda hn: min(host_to_ranks[hn]))

            new_host_to_goal_rank = {hn: i for i, hn in enumerate(host_order)}

            # Determine stable GPU ID order: per host, ranks sorted ascending.
            old_gpu_to_new_gpu = {}
            next_gpu = 0
            for hn in host_order:
                for r in sorted(host_to_ranks[hn]):
                    ri = ref_rank_info[str(r)]
                    old_gpu = int(ri["gpuId"])
                    if old_gpu not in old_gpu_to_new_gpu:
                        old_gpu_to_new_gpu[old_gpu] = next_gpu
                        next_gpu += 1

            # Any remaining gpuIds (from smaller comms) are appended in sorted old_gpu order.
            all_old_gpus = set()
            for gr in nccl_events.keys():
                all_old_gpus |= set(int(g) for g in nccl_events[gr].keys())
            for old_gpu in sorted(all_old_gpus):
                if old_gpu not in old_gpu_to_new_gpu:
                    old_gpu_to_new_gpu[old_gpu] = next_gpu
                    next_gpu += 1

            # Remap host->goal_rank
            HostName_To_GoalRank = dict(new_host_to_goal_rank)
            GoalRank_To_NumOfGPUs = {i: len(host_to_ranks[hn]) for hn, i in new_host_to_goal_rank.items()}

            # Remap per-goal-rank structures keyed by the *old* goal_rank
            old_goal_rank_to_host = {gr: hn for hn, gr in HostName_To_GoalRank.items()}
            # old_goal_rank_to_host is now new mapping (post overwrite); recover host_name from existing
            # trace-derived structures instead: build a host map from ref comm.
            old_goal_rank_to_host = {}
            for rank_str, ri in ref_rank_info.items():
                # old goal_rank (from parsing) is stored in rank info as 'goal_rank'
                old_goal_rank_to_host[int(ri["goal_rank"])] = ri["host_name"]

            def remap_goal_rank(old_gr: int) -> int:
                hn = old_goal_rank_to_host.get(old_gr)
                if hn is None:
                    return old_gr
                return new_host_to_goal_rank[hn]

            def remap_gpu(old_gpu: int) -> int:
                return old_gpu_to_new_gpu[int(old_gpu)]

            def remap_nested_by_goal_rank_and_gpu(src):
                dst = {}
                for old_gr, gpus in src.items():
                    new_gr = remap_goal_rank(int(old_gr))
                    if new_gr not in dst:
                        dst[new_gr] = {}
                    for old_gpu, payload in gpus.items():
                        dst[new_gr][remap_gpu(int(old_gpu))] = payload
                return dst

            nccl_events = remap_nested_by_goal_rank_and_gpu(nccl_events)
            cupti_kernel_results = remap_nested_by_goal_rank_and_gpu(cupti_kernel_results)
            comm_init_events = remap_nested_by_goal_rank_and_gpu(comm_init_events)
            events_counter = remap_nested_by_goal_rank_and_gpu(events_counter)

            # Remap profile_interval (keyed by gpuId)
            profile_interval = {remap_gpu(int(gpu)): v for gpu, v in profile_interval.items()}

            # Remap comm_info for all comms
            for commId, ci in comm_info.items():
                rank_info = ci.get("rank_To_rankInfo", {})
                new_gpuId_to_rank = {}
                for rank_str, ri in rank_info.items():
                    hn = ri["host_name"]
                    ri["goal_rank"] = new_host_to_goal_rank.get(hn, int(ri["goal_rank"]))
                    ri["gpuId"] = remap_gpu(int(ri["gpuId"]))
                    new_gpuId_to_rank[int(ri["gpuId"])] = rank_str
                ci["gpuId_To_rank"] = new_gpuId_to_rank

    # Optional: rotate goal-rank numbering (debugging / compatibility knob).
    # This is useful if downstream tooling implicitly assumes a particular host ordering.
    #
    # Example: export ATLAHS_GOAL_RANK_ROTATE=-1  (rotate right by 1)
    rotate_str = os.environ.get("ATLAHS_GOAL_RANK_ROTATE", "0").strip()
    try:
        rotate = int(rotate_str)
    except ValueError:
        rotate = 0

    num_goal_ranks = len(HostName_To_GoalRank)
    if rotate != 0 and num_goal_ranks > 0:
        rotate %= num_goal_ranks

        def rotate_rank(r: int) -> int:
            return (r + rotate) % num_goal_ranks

        HostName_To_GoalRank = {hn: rotate_rank(gr) for hn, gr in HostName_To_GoalRank.items()}
        GoalRank_To_NumOfGPUs = {rotate_rank(gr): n for gr, n in GoalRank_To_NumOfGPUs.items()}
        comm_init_events = {rotate_rank(gr): v for gr, v in comm_init_events.items()}
        nccl_events = {rotate_rank(gr): v for gr, v in nccl_events.items()}
        cupti_kernel_results = {rotate_rank(gr): v for gr, v in cupti_kernel_results.items()}
        events_counter = {rotate_rank(gr): v for gr, v in events_counter.items()}

        for commId, ci in comm_info.items():
            rank_info = ci.get("rank_To_rankInfo", {})
            for _, ri in rank_info.items():
                if "goal_rank" in ri:
                    ri["goal_rank"] = rotate_rank(int(ri["goal_rank"]))

    return comm_init_events, nccl_events, cupti_kernel_results, comm_info, HostName_To_GoalRank, profile_interval


def merge_stream_if_no_overlap(nccl_events, cupti_kernel_results):
    """
    Iterates through the NCCL events collected on all streams of a GPU and merges them if
    no overlap is detected.
    Returns the original nccl_events if no overlap is detected, otherwise returns the merged events.
    """
    merged_nccl_events = {}
    merged_cupti_kernel_results = {}

    for goal_rank in sorted(nccl_events.keys(), key=int):
        gpu_events = nccl_events[goal_rank]
        merged_nccl_events[goal_rank] = {}
        merged_cupti_kernel_results[goal_rank] = {}
        for gpuId in sorted(gpu_events.keys(), key=int):
            merged_nccl_events[goal_rank][gpuId] = {}
            merged_cupti_kernel_results[goal_rank][gpuId] = {}
            tmp_nccl_events = []
            tmp_cupti_kernel_results = []

            for streamId in sorted(nccl_events[goal_rank][gpuId].keys(), key=int):
                stream_events = nccl_events[goal_rank][gpuId][streamId]
                tmp_nccl_events.extend(stream_events)
            
            for streamId in sorted(cupti_kernel_results[goal_rank][gpuId].keys(), key=int):
                stream_events = cupti_kernel_results[goal_rank][gpuId][streamId]
                tmp_cupti_kernel_results.extend(stream_events)
            
            tmp_nccl_events.sort(key=lambda x: x['ts_start'])
            tmp_cupti_kernel_results.sort(key=lambda x: x['ts_gpu_start'])
            assert len(tmp_nccl_events) == sum([len(v) for v in nccl_events[goal_rank][gpuId].values()]), 'Different number of events in nccl_events'
            assert len(tmp_cupti_kernel_results) == sum([len(v) for v in cupti_kernel_results[goal_rank][gpuId].values()]), 'Different number of events in cupti_kernel_results'

            # Checks if there is any overlap between the events.
            # Some traces (especially after post-filtering or late-tracing recovery) may have
            # a different number of NCCL NVTX intervals vs CUPTI kernels. In that case we
            # cannot safely align them for stream-merging; keep the original per-stream events.
            if len(tmp_nccl_events) != len(tmp_cupti_kernel_results):
                if os.environ.get("ATLAHS_FORCE_STREAM_MERGE_ON_MISMATCH", "0").strip() != "1":
                    print(f"[WARN] GPU {gpuId} on goal rank {goal_rank}: different number of NCCL vs CUPTI events ({len(tmp_nccl_events)} != {len(tmp_cupti_kernel_results)}); skipping stream merge")
                    return nccl_events, cupti_kernel_results
                print(f"[WARN] GPU {gpuId} on goal rank {goal_rank}: NCCL vs CUPTI event count mismatch ({len(tmp_nccl_events)} != {len(tmp_cupti_kernel_results)}); forcing stream merge (ATLAHS_FORCE_STREAM_MERGE_ON_MISMATCH=1)")

            for i in range(len(tmp_nccl_events) - 1):
                if tmp_nccl_events[i]['ts_end'] > tmp_nccl_events[i+1]['ts_start'] or \
                    (i < len(tmp_cupti_kernel_results) - 1 and tmp_cupti_kernel_results[i]['ts_gpu_end'] > tmp_cupti_kernel_results[i+1]['ts_gpu_start']):
                    print(f"[INFO] Overlap detected for GPU {gpuId} on goal rank {goal_rank}, not merging streams")
                    return nccl_events, cupti_kernel_results

            merged_nccl_events[goal_rank][gpuId][0] = tmp_nccl_events
            merged_cupti_kernel_results[goal_rank][gpuId][0] = tmp_cupti_kernel_results
            print(f"[INFO] No overlap detected for GPU {gpuId} on goal rank {goal_rank}")
            print(f"[INFO] Streams from GPU {gpuId} have been merged into a single stream, number of events: {len(tmp_nccl_events)}")
    
    return merged_nccl_events, merged_cupti_kernel_results
