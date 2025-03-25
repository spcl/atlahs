#include "atlahs_htsim_api.h"
#include "eqds.h"
#include "atlahs_event.h"
#include "fat_tree_topology.h"
#include "logsim-interface.h"


void AtlahsHtsimApi::Send(const SendEvent &event) {
    std::cout << "AtlahsHtsimApi: Sending event" << std::endl;

    // TODO: Move this stuff to a CreateConnection function inside UEC. 
    // TODO: Support different tranports, not just UEC
    /* if (_uecRtxScanner == NULL) {
        _uecRtxScanner = new UecRtxTimerScanner(50000, *_eventlist);
    }

    // This is updated inside UEC if it doesn't fit the default values
    uint64_t rtt = BASE_RTT_MODERN * 1000;
    uint64_t bdp = BDP_MODERN_UEC;


    UecSrc *uecSrc = new UecSrc(NULL, NULL, *_eventlist, rtt, bdp, 0, 6);
    uecSrc->setName("uec_" + std::to_string(event.getFrom()) + "_" + std::to_string(event.getTo()));
    uecSrc->_atlahs_api = this;
    //uecSrc->set_flow_over_hook(std::bind(&LogSimInterface::flow_over, this, std::placeholders::_1));
    uecSrc->from = event.getFrom();
    uecSrc->to = event.getTo();
    uecSrc->tag = event.getTag();
    uecSrc->send_size = event.getSizeBytes();
    std::string to_hash = std::to_string(event.getFrom()) + "@" + std::to_string(event.getTo()) + "@" + std::to_string(event.getTag());
    UecSink *uecSink = new UecSink();
    uecSink->setName("uec_sink_Rand");
    uecSink->from = event.getFrom();
    uecSink->to = event.getTo();
    uecSink->tag = event.getTag();

    uecSrc->setFlowSize(event.getSizeBytes());
    uecSrc->set_dst(event.getTo());
    uecSink->set_src(event.getFrom());

    uecSrc->set_paths(256);
    uecSink->set_paths(256);

    Route *srctotor = new Route();
    Route *dsttotor = new Route();

    int from = event.getFrom();
    int to = event.getTo();

    if (_topo != NULL) {
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)]);
        srctotor->push_back(_topo->pipes_ns_nlp[from][_topo->HOST_POD_SWITCH(from)]);
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)]->getRemoteEndpoint());

        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)]);
        dsttotor->push_back(_topo->pipes_ns_nlp[to][_topo->HOST_POD_SWITCH(to)]);
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)]->getRemoteEndpoint());
    } 

    uecSrc->connect(srctotor, dsttotor, *uecSink, 0);

    if (_topo != NULL) {
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(to)]);

        _topo->switches_lp[_topo->HOST_POD_SWITCH(from)]->addHostPort(from, uecSrc->flow_id(), uecSrc);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(to)]->addHostPort(to, uecSrc->flow_id(), uecSink);
    } */
}

void AtlahsHtsimApi::Recv(const RecvEvent &event) {
    std::cout << "AtlahsHtsimApi: Receiving event" << std::endl;
}

void AtlahsHtsimApi::Calc(const ComputeAtlahsEvent &event) {
    std::cout << "AtlahsHtsimApi: Performing calculation on event" << std::endl;
}

void AtlahsHtsimApi::Setup() {
    std::cout << "AtlahsHtsimApi: Setting up simulator. Currently Dummy in this version." << std::endl;
}

void AtlahsHtsimApi::EventFinished(const EventOver &event) {
    std::cout << "AtlahsHtsimApi: Event is over" << std::endl;

    if (AtlahsEventType::SEND_EVENT_OVER == event.getEventType()) {
        _logsim_interface->flow_over(*(event.getPacket()));
    } else if (AtlahsEventType::COMPUTE_EVENT_OVER == event.getEventType()) {
        _logsim_interface->compute_over(1);
    } else {
        abort();
    }
    
}