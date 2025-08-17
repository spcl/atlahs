#include "atlahs_htsim_api.h"
#include "eqds.h"
#include "atlahs_event.h"
#include "fat_tree_topology.h"
#include "logsim-interface.h"
#include "../lgs/LogGOPSim.hpp"
    
bool AtlahsHtsimApi::llama_rand = false;

void AtlahsHtsimApi::Send(const SendEvent &event, graph_node_properties elem) {
    //std::cout << "AtlahsHtsimApi: Sending event" << std::endl;

    int to = event.getTo();
    int from = event.getFrom();
    int tag = event.getTag();
    int size = event.getSizeBytes();
    size = size * 1;

    if (getNumberNic() > 1) {
        from = getHtsimNodeNumber(from, elem.nic);
        to = getHtsimNodeNumber(to, elem.nic);
    }

    // Temporary solution
    if (llama_rand) {
        if (from >= 4) {
            from = from + 12;
            to = to + 12;
        }
    }
    
    if (from == to) {
        std::cerr << "Error: Send event from and to the same node" << std::endl;
        exit(0);
    }

    if (_logsim_interface->get_protocol() == EQDS_PROTOCOL) {
        EqdsSrc* eqds_src;
        EqdsSink* eqds_snk;
        eqds_src = new EqdsSrc(NULL, *_eventlist, *nics.at(from));
        //eqds_src->setCwnd(initial_cwnd);
        //eqds_srcs.push_back(eqds_src);
        eqds_src->setDst(to);
        /* printf("Setting up connection from %d to %d\n", from, to);
        printf("Size Pacers %d\n", pacers.size());
        printf("Size NICs %d\n", nics.size()); */
        eqds_snk = new EqdsSink(NULL,pacersEQDS[to],*nics.at(to));
        eqds_src->setName("Eqds_" + ntoa(from) + "_" + ntoa(to));
        eqds_snk->setSrc(from);
                        
        eqds_snk->setName("Eqds_sink_" + ntoa(from) + "_" + ntoa(to));
        eqds_src->setFlowsize(size);

        Route* srctotor = new Route();
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]);
        srctotor->push_back(_topo->pipes_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]);
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]->getRemoteEndpoint());

        Route* dsttotor = new Route();
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]);
        dsttotor->push_back(_topo->pipes_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]);
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]->getRemoteEndpoint());

        eqds_src->from = from;
        eqds_src->to = to;
        eqds_src->tag = tag;
        eqds_src->_atlahs_api = this;

        graph_node_properties* node_copy = new graph_node_properties(elem);
        eqds_src->lgs_node = node_copy;

        eqds_src->connect(*srctotor, *dsttotor, *eqds_snk, _eventlist->now());
        //eqds_src->setPaths(path_entropy_size);
        //eqds_snk->setPaths(path_entropy_size);

        //register src and snk to receive packets from their respective TORs. 
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(from)]->addHostPort(from,eqds_snk->flowId(),eqds_src);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(to)]->addHostPort(to,eqds_src->flowId(),eqds_snk);
    } else if (_logsim_interface->get_protocol() == NDP_PROTOCOL) {
        NdpSrc* ndpSrc;
        NdpSink* ndpSnk;
        ndpSrc = new NdpSrc(NULL, NULL, *_eventlist, false);
        ndpSrc->setCwnd(50*Packet::data_packet_size());
        ndpSrc->set_dst(to);
        ndpSrc->set_flowsize(size);

        ndpSnk = new NdpSink(pacersNDP[to]);
        ndpSnk->set_src(from);
                        
        ndpSnk->setName("ndp_sink_" + ntoa(from) + "_" + ntoa(to));
        ndpSrc->setName("ndp_" + ntoa(from) + "_" + ntoa(to));

        Route* srctotor = new Route();
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]);
        srctotor->push_back(_topo->pipes_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]);
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]->getRemoteEndpoint());

        Route* dsttotor = new Route();
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]);
        dsttotor->push_back(_topo->pipes_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]);
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]->getRemoteEndpoint());

        ndpSrc->from = from;
        ndpSrc->to = to;
        ndpSrc->tag = tag;
        ndpSnk->from_sink = from;
        ndpSnk->to_sink = to;
        ndpSnk->tag_sink = tag;
        ndpSrc->_atlahs_api = this;

        graph_node_properties* node_copy = new graph_node_properties(elem);
        ndpSrc->lgs_node = node_copy;
        ndpSrc->connect(srctotor, dsttotor, *ndpSnk, _eventlist->now());
        ndpSrc->set_paths(128);
        ndpSnk->set_paths(128);

        //register src and snk to receive packets from their respective TORs. 
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(from)]->addHostPort(from,ndpSrc->flow_id(),ndpSrc);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(to)]->addHostPort(to,ndpSrc->flow_id(),ndpSnk);
    } else if (_logsim_interface->get_protocol() == SENDER_PROTOCOL) { 

        UecSrc *uecSrc = new UecSrc(NULL, NULL, *_eventlist, getSenderRtt(), getSenderBdp(), 100, 6);

        uecSrc->setFlowSize(size);

        uecSrc->setName("uec_" + std::to_string(from) + "_" + std::to_string(to));
        uecSrc->from = from;
        uecSrc->to = to;
        uecSrc->tag = tag;
        uecSrc->send_size = size;
        uecSrc->_atlahs_api = this;

        UecSink *uecSink = new UecSink();
        uecSink->setName("uec_sink_Rand");
        uecSink->from_sink = from;
        uecSink->to_sink = to;
        uecSink->tag_sink = tag;

        uecSrc->set_dst(to);
        uecSink->set_src(from);

        Route* srctotor = new Route();
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]);
        srctotor->push_back(_topo->pipes_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]);
        srctotor->push_back(_topo->queues_ns_nlp[from][_topo->HOST_POD_SWITCH(from)][0]->getRemoteEndpoint());

        Route* dsttotor = new Route();
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]);
        dsttotor->push_back(_topo->pipes_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]);
        dsttotor->push_back(_topo->queues_ns_nlp[to][_topo->HOST_POD_SWITCH(to)][0]->getRemoteEndpoint());

        graph_node_properties* node_copy = new graph_node_properties(elem);
        uecSrc->lgs_node = node_copy;
        uecSrc->connect(srctotor, dsttotor, *uecSink, _eventlist->now());

        uecSrc->set_paths(128);
        uecSink->set_paths(128);

        //register src and snk to receive packets from their respective TORs. 
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        assert(_topo->switches_lp[_topo->HOST_POD_SWITCH(from)]);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(from)]->addHostPort(from,uecSrc->flow_id(),uecSrc);
        _topo->switches_lp[_topo->HOST_POD_SWITCH(to)]->addHostPort(to,uecSrc->flow_id(),uecSink);

    }
    // TODO: Move this stuff to a CreateConnection function inside UEC. 
    // TODO: Support different tranports, not just UEC
}

void AtlahsHtsimApi::Recv(const RecvEvent &event) {
    // No Op for HTSIM
}

void AtlahsHtsimApi::Calc(const ComputeAtlahsEvent &event) {
    // Done Directly in lgs_interface for now
}

void AtlahsHtsimApi::Setup() {
    printf("No of nodes %d\n", total_nodes);

    if (_logsim_interface->get_protocol() == EQDS_PROTOCOL) {
        for (size_t ix = 0; ix < total_nodes; ix++){
            printf("Setting up node %d\n", ix);
            pacersEQDS.push_back(new EqdsPullPacer(linkspeed, 0.99, EqdsSrc::_mtu, *_eventlist));   
            nics.push_back(new EqdsNIC(*_eventlist, linkspeed));
        }
    } else if (_logsim_interface->get_protocol() == NDP_PROTOCOL) {
        for (size_t ix = 0; ix < total_nodes; ix++)
            pacersNDP.push_back(new NdpPullPacer(*_eventlist,  linkspeed, 0.99));   
    }
    
}

void AtlahsHtsimApi::EventFinished(const EventOver &event) {
    //std::cout << "AtlahsHtsimApi: Event is over" << std::endl;

    if (AtlahsEventType::SEND_EVENT_OVER == event.getEventType()) {
        //_logsim_interface->flow_over(*(event.getPacket()));
        _logsim_interface->flow_over(event);
    } else if (AtlahsEventType::COMPUTE_EVENT_OVER == event.getEventType()) {
        _logsim_interface->compute_over(1);
    } else {
        abort();
    }
}