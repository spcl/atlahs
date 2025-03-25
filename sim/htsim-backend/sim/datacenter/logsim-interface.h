/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef LOGSIM_INTERFACE
#define LOGSIM_INTERFACE

#include "compute_event.h"
#include "eventlist.h"
#include "fat_tree_topology.h"
#include "lgs/logsim.h"
#include "eqds.h"
#include "atlahs_htsim_api.h"
#include "atlahs_event.h"
#include <string>
#include <unordered_map>

class graph_node_properties;
class NdpRtxTimerScanner;
class NdpPullPacer;
class Topology;
class UecRtxTimerScanner;
class SwiftTrimmingRtxTimerScanner;
class UecDropRtxTimerScanner;

enum ProtocolName {
    NDP_PROTOCOL,
    UEC_PROTOCOL,
    SWIFT_PROTOCOL,
    UEC_DROP_PROTOCOL,
    EQDS_PROTOCOL
};

/* ... */
class MsgInfo {
    // Access specifier
  public:
    // Data  Members
    int total_bytes_msg;
    int bytes_left_to_recv;
    int identifier;
    u_int64_t start_time;
    int offset;
    int to_parse;
};

//

class LogSimInterface {
  public:
    LogSimInterface();
    LogSimInterface(EqdsLogger *logger, TrafficLogger *pktLogger,
                    EventList &eventList, FatTreeTopology *,
                    std::vector<const Route *> ***);
    std::unordered_map<std::string, MsgInfo> active_sends;
    int compute_started = 0;
    void htsim_schedule(u_int32_t, int, int, int, u_int64_t, int);
    void send_event(int from, int to, int size, int tag,
                    u_int64_t start_time_event);
    void execute_compute(graph_node_properties elem, int p);
    void set_protocol(ProtocolName name) { _protocolName = name; };
    void set_cwd(int cwd);
    void setReuse(bool reuse) { _use_good_entropies = reuse; };
    void setIgnoreEcnAck(bool ignore_ecn_ack) {
        _ignore_ecn_ack = ignore_ecn_ack;
    };
    void setIgnoreEcnData(bool ignore_ecn_data) {
        _ignore_ecn_data = ignore_ecn_data;
    };
    void setNumberEntropies(int num_entropies) {
        _num_entropies = num_entropies;
    };
    void setNumberPaths(int num_paths) { path_entropy_size = num_paths; };

    void set_queue_size(int queue_size) { _queuesize = queue_size; };
    std::unordered_map<std::string, MsgInfo> get_active_sends();
    void update_active_map(std::string, int);
    bool all_sends_delivered();
    void ns3_terminate(int64_t &current_time);
    void flow_over(const Packet &);
    void compute_over(int);
    graph_node_properties htsim_simulate_until(u_int64_t until);
    void update_latest_receive(graph_node_properties *recv_op);
    void reset_latest_receive();
    void terminate_sim();
    int track_times[8192] = {0};
    std::uint64_t htsim_time = 0;

    AtlahsHtsimApi* htsim_api;

  private:
    bool debug_prints = false;
    TrafficLogger *_flow;
    EqdsLogger *_logger;
    EventList *_eventlist;
    FatTreeTopology *_topo = NULL;
    std::vector<const Route *> ***_netPaths;
    int _cwd;
    ComputeEvent *compute_events_handler;
    graph_node_properties *_latest_recv;
    bool compute_if_finished = false;
    ProtocolName _protocolName;
    int _queuesize;
    std::unordered_map<int, NdpPullPacer *> _puller_map;
    bool _use_good_entropies;
    bool _ignore_ecn_ack;
    bool _ignore_ecn_data;
    int _num_entropies;
    int path_entropy_size = 256;
};

int start_lgs(std::string, LogSimInterface &);
#endif /* LOGSIM_HELPER_H */
