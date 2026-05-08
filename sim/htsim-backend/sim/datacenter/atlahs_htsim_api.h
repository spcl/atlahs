#ifndef ATLAHS_HTSIM_API_H
#define ATLAHS_HTSIM_API_H

#include "atlahs_api.h"
#include <iostream>
#include <functional>
#include "../compute_event.h"
#include "../null_event.h"
#include "atlahs_event.h"

// Forward declarations
class EventList;
class UecRtxTimerScanner;
class FatTreeTopology;
class LogSimInterface;
class EventOver;
class EqdsPullPacer;
class EqdsNIC;
class NdpPullPacer;
//class ComputeEvent;

// Added FlowInfo class
class FlowInfo {
public:
    // Default constructor
    FlowInfo() 
        : flowStartTime(0), flowEndTime(0), completionTime(0), flowSize(0), numNacks(0), finalCwnd(0) 
    {}

    // Parameterized constructor
    FlowInfo(simtime_picosec start, simtime_picosec end, simtime_picosec completion,
                uint64_t size, uint64_t nacks, uint64_t cwnd)
        : flowStartTime(start), flowEndTime(end), completionTime(completion),
            flowSize(size), numNacks(nacks), finalCwnd(cwnd)
    {}

    simtime_picosec flowStartTime;
    simtime_picosec flowEndTime;
    simtime_picosec completionTime;
    uint64_t flowSize;
    uint64_t numNacks;
    uint64_t finalCwnd;
};

class AtlahsHtsimApi : public AtlahsApi {
public:
    enum class GoalRankMapping {
        GpuRank,
        UniqueNic,
    };

    AtlahsHtsimApi() = default;
    virtual ~AtlahsHtsimApi() = default;
    
    virtual void Send(const SendEvent &event, graph_node_properties node) override;
    virtual void Recv(const RecvEvent &event) override;
    virtual void Calc(const ComputeAtlahsEvent &event) override;
    virtual void Setup() override;
    virtual void EventFinished(const EventOver &event) override;

    // Getter and setter for EventList
    void setEventList(EventList* eventlist) { _eventlist = eventlist; }
    EventList* getEventList() const { return _eventlist; }
    
    // Getter and setter for UecRtxTimerScanner
    void setUecRtxScanner(UecRtxTimerScanner* scanner) { _uecRtxScanner = scanner; }
    UecRtxTimerScanner* getUecRtxScanner() const { return _uecRtxScanner; }
    
    // Getter and setter for FatTreeTopology
    void setTopology(FatTreeTopology* topo) { _topo = topo; }
    FatTreeTopology* getTopology() const { return _topo; }

    // Getter and setter for LogSimInterface
    void setLogSimInterface(LogSimInterface* logsim_interface) { _logsim_interface = logsim_interface; }
    LogSimInterface* getLogSimInterface() const { return _logsim_interface; }

    // Getter and setter for ComputeEvent
    void setComputeEvent(ComputeEvent* compute_event) { 
        compute_events_handler = compute_event; 
        compute_events_handler->set_compute_over_hook(
            std::bind(&AtlahsHtsimApi::compute_over_intermediate, this, std::placeholders::_1));
    }
    ComputeEvent* getComputeEvent() const { return compute_events_handler; }

    void setNullEvent(NullEvent* Null_event) { 
        null_events_handler = Null_event; 
        null_events_handler->set_null_over_hook(
            std::bind(&AtlahsHtsimApi::null_over_intermediate, this, std::placeholders::_1));
    }
    NullEvent* getNullEvent() const { return null_events_handler; }


    void compute_over_intermediate(int i) {
        EventOver event;
        event.event_type = AtlahsEventType::COMPUTE_EVENT_OVER;
        this->EventFinished(event);
        return;
    }


    void null_over_intermediate(int i) {
        EventOver event;
        event.event_type = AtlahsEventType::COMPUTE_EVENT_OVER;
        this->EventFinished(event);
        return;
    }

    void setSenderCwnd(int cwnd) { sender_cwnd = cwnd; }
    int getSenderCwnd() const { return sender_cwnd; }
    
    void setSenderRtt(int rtt) { sender_rtt = rtt; }
    int getSenderRtt() const { return sender_rtt; }
    
    void setSenderBdp(int bdp) { sender_bdp = bdp; }
    int getSenderBdp() const { return sender_bdp; }

    void setNumberNic(int nic) { number_nics = nic; }
    int getNumberNic() const { return number_nics; }

    void setGoalRankMapping(GoalRankMapping mapping) { goal_rank_mapping = mapping; }
    GoalRankMapping getGoalRankMapping() const { return goal_rank_mapping; }

    void setGoalRankMappingFromBinaryHeader(uint32_t rank_count, int cpu_count, int nic_count) {
        setNumberNic(nic_count);

        // The binary GOAL format does not store the generator version. Infer
        // the layout from the parsed schedule header instead: V1 unique-nic
        // files keep ranks at node granularity and use several NICs per rank,
        // while V2 files keep ranks at GPU/HTSIM-node granularity.
        const bool looks_like_v2_gpu_rank =
            nic_count == 1 ||
            (nic_count == 2 && rank_count > static_cast<uint32_t>(nic_count) && cpu_count <= 8);
        goal_rank_mapping =
            looks_like_v2_gpu_rank ? GoalRankMapping::GpuRank : GoalRankMapping::UniqueNic;
    }

    bool usesUniqueNicRankMapping() const {
        return goal_rank_mapping == GoalRankMapping::UniqueNic;
    }

    const char* getGoalRankMappingName() const {
        switch (goal_rank_mapping) {
            case GoalRankMapping::UniqueNic:
                return "unique-nic";
            case GoalRankMapping::GpuRank:
                return "gpu-rank";
        }
        return "unknown";
    }

    void setNumberNacks(int nacks) { number_of_nacks += nacks; }
    uint64_t getNumberNacks() const { return number_of_nacks; }

    simtime_picosec getGlobalTimePs() const { return _eventlist->now(); }
    simtime_picosec getGlobalTimeNs() const { return _eventlist->now() / 1000; }

    int getHtsimNodeNumber(int lgs_host, int lgs_nic) {
        return usesUniqueNicRankMapping() ? lgs_host * number_nics + lgs_nic : lgs_host;
    }

    linkspeed_bps linkspeed; // TO DO
    int linkspeed_gbps = 100; // TO DO
    double htsim_G; // TO DO
    int total_nodes; // TO DO
    bool send_done_return_control = false; // TO DO
    std::vector<FlowInfo> flowInfos;
    bool print_stats_flows = false;
    bool static llama_rand;
    
private:
    EventList* _eventlist = nullptr;
    UecRtxTimerScanner* _uecRtxScanner = nullptr;
    FatTreeTopology* _topo = nullptr;
    LogSimInterface* _logsim_interface = nullptr;
    ComputeEvent *compute_events_handler = nullptr;
    NullEvent *null_events_handler = nullptr;

    // LGS Specific
    int number_nics = 1;
    GoalRankMapping goal_rank_mapping = GoalRankMapping::GpuRank;

    // EQDS Specific 
    vector<EqdsPullPacer*> pacersEQDS;
    vector<EqdsNIC*> nics;
    int initial_cwnd = 100000000;

    // NDP Specific
    vector<NdpPullPacer*> pacersNDP;

    // Sender Specific
    int sender_cwnd = 0;
    int sender_rtt = 0;
    int sender_bdp = 0;

    // Networking Stats
    uint64_t number_of_nacks = 0;
};

#endif // ATLAHS_HTSIM_API_H
