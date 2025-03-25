/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "logsim-interface.h"
#include "lgs/LogGOPSim.hpp"
#include "lgs/Network.hpp"
#include "lgs/Noise.hpp"
#include "lgs/Parser.hpp"
#include "lgs/TimelineVisualization.hpp"
#include "lgs/cmdline.h"
#include "main.h"
#include "topology.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <regex>
#include <string>
#include <utility>
#include <unordered_set>
/*#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS*/

#define DEBUG_PRINT 0

static bool print = false;



LogSimInterface::LogSimInterface() {}

LogSimInterface::LogSimInterface(EqdsLogger *logger, TrafficLogger *pktLogger, EventList &eventList,
                                 FatTreeTopology *topo, std::vector<const Route *> ***routes) {
    _logger = logger;
    _flow = pktLogger;
    _eventlist = &eventList;
    _topo = topo;
    _netPaths = routes;
    _latest_recv = new graph_node_properties();
    if (compute_events_handler == NULL) {
        compute_events_handler = new ComputeEvent(*_eventlist);
        compute_events_handler->set_compute_over_hook(
                std::bind(&LogSimInterface::compute_over, this, std::placeholders::_1));
    }
}

void LogSimInterface::set_cwd(int cwd) { _cwd = cwd; }

void LogSimInterface::htsim_schedule(u_int32_t host, int to, int size, int tag, u_int64_t start_time_event,
                                     int my_offset) {
    // Send event to htsim for actual send
    send_event(host, to, size, tag, start_time_event);

    // Save Event for internal tracking
    std::string to_hash = std::to_string(host) + "@" + std::to_string(to) + "@" + std::to_string(tag);
    /* printf("Scheduling Event (%s) of size %d from %d to %d tag %d start_tiem "
           "%lu - Time is %lu\n ",
           to_hash.c_str(), size, host, to, tag, start_time_event * 1000, GLOBAL_TIME); */
    MsgInfo entry;
    entry.start_time = start_time_event * 1;
    entry.total_bytes_msg = size;
    entry.offset = my_offset;
    entry.bytes_left_to_recv = size;
    entry.to_parse = 42;
    active_sends[to_hash] = entry;
}

void LogSimInterface::execute_compute(graph_node_properties comp_elem, int size_p) {
    if (_protocolName == UEC_PROTOCOL) {
        compute_events_handler->setCompute(comp_elem.size);

        ComputeAtlahsEvent *compute_event = new ComputeAtlahsEvent(comp_elem.size);
        htsim_api->Calc(*compute_event);
    }
}

void LogSimInterface::send_event(int from, int to, int size, int tag, u_int64_t start_time_event) {

    // Testing New 
    SendEvent* event = new SendEvent(from, to, size, tag, start_time_event);    
    htsim_api->Send(*event);

    /* printf("LGS Send Event - Time %lu - Host %d - Dst %d - Tag %d - Size %d - "
           "StartTime %d\n",
           GLOBAL_TIME / 1000, from, to, tag, size, start_time_event); */

    return;
}

void LogSimInterface::update_active_map(std::string to_hash, int size) {

    // Check that the flow actually exists
    active_sends[to_hash].bytes_left_to_recv = active_sends[to_hash].bytes_left_to_recv - Packet::data_packet_size();
    if (active_sends[to_hash].bytes_left_to_recv <= 0) {
    }
}

bool LogSimInterface::all_sends_delivered() { return active_sends.size() == 0; }

void LogSimInterface::flow_over(const Packet &p) {

    // Get Unique Hash
    std::string to_hash = std::to_string(p.from) + "@" + std::to_string(p.to) + "@" + std::to_string(p.tag);
    // active_sends[to_hash].bytes_left_to_recv = 0;
    /* printf("Flow Finished %d@%d@%d at %lu\n", p.from, p.to, p.tag, GLOBAL_TIME); */

    // Here we have received a message fully, we need to give control back to
    // LGS
    _latest_recv = new graph_node_properties();
    _latest_recv->updated = true;
    _latest_recv->tag = p.tag;
    _latest_recv->type = OP_MSG;
    _latest_recv->target = p.from;
    _latest_recv->host = p.to;
    _latest_recv->starttime = htsim_api->getGlobalTimeNs();
    _latest_recv->time = htsim_api->getGlobalTimeNs();
    _latest_recv->size = active_sends[to_hash].total_bytes_msg;
    _latest_recv->offset = active_sends[to_hash].offset;
    _latest_recv->proc = 0;
    _latest_recv->nic = 0;

    active_sends.erase(to_hash);
    if (_protocolName == EQDS_PROTOCOL) {
        /* connection_log.erase(to_hash); */
    }
    return;
}

void LogSimInterface::compute_over(int i) {
    /*  printf("Compute is over, time is %lu vs htsim_time %lu\n", GLOBAL_TIME, htsim_time); */
    compute_started--;
    htsim_time = htsim_api->getGlobalTimeNs();
    compute_if_finished = true;
    return;
}

void LogSimInterface::reset_latest_receive() { _latest_recv->updated = false; }

void LogSimInterface::terminate_sim() {
    /* if (_protocolName == EQDS_PROTOCOL) {
        for (std::size_t i = 0; i < _uecSrcVector.size(); ++i) {
            delete _uecSrcVector[i];
        }
    } */
}

graph_node_properties LogSimInterface::htsim_simulate_until(u_int64_t until) {


    while (_eventlist->doNextEvent()) {

        if (_latest_recv->updated) {
            break;
        }

        if (compute_if_finished) {
            compute_if_finished = false;
            break;
        }
    }

    return *_latest_recv;
}

int start_lgs(std::string filename_goal, LogSimInterface &lgs) {
    LogSimInterface *lgs_interface = &lgs;

    //filename_goal = ("sim/lgs/input/" + filename_goal);


    // Time Inside LGS
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    // auto start = high_resolution_clock::now();
    std::chrono::milliseconds global_time = std::chrono::milliseconds::zero();

    //  End Temp Only

    #ifdef STRICT_ORDER
    btime_t aqtime=0; 
    uint64_t num_reinserts_o = 0;
    uint64_t num_reinserts_g = 0;
    uint64_t num_reinserts_net = 0;
  #endif

  #ifndef LIST_MATCH
    if( qstat_given ){
      printf("WARNING: --qstat option provided, but LogGOPSim was compiled with LIST_MATCH;\n"
             "         statistics on match queue behavior are NOT valid.\n"); 
    }
  #endif
  
    // read input parameters
    const int o = 1500;
    const int O = 0;
    const int g = 1000;
    const int L = 2500;
    const double G = 6;
    print = false;
    const uint32_t S = 65535;
    lgs_interface->htsim_time = 0;

    Parser parser(filename_goal, true);
    //   std::cout << "[DEBUG] Comm dep file: " << comm_dep_file_arg << std::endl;
    // Network net(&args_info);
  
    const uint p = parser.schedules.size();
    const int ncpus = parser.GetNumCPU();
    const int nnics = parser.GetNumNIC();

    bool comm_dep_file_arg = false;
  
    printf("size: %i (%i CPUs, %i NICs); L=%i, o=%i g=%i, G=%f, O=%i, P=%i, S=%u\n", 
           p, ncpus, nnics, L, o, g, G, O, p, S);
    
    // DATA structures for storing MPI matching statistics
    std::vector<int> rq_max(0);
    std::vector<int> uq_max(0); 
  
    std::vector< std::vector< std::pair<int,btime_t> > > rq_matches(0);
    std::vector< std::vector< std::pair<int,btime_t> > > uq_matches(0);
  
    std::vector< std::vector< std::pair<int,btime_t> > > rq_misses(0);
    std::vector< std::vector< std::pair<int,btime_t> > > uq_misses(0);
  
    std::vector< std::vector<btime_t> > rq_times(0);
    std::vector< std::vector<btime_t> > uq_times(0); 
  
    // Data structure for keeping track of communication dependencies
    // Stores a vector of tuples each containing four elements:
    // (source rank, source offset, dest rank, dest offset)
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> comm_deps;
  
    if( false ){ // TO DO
      // Initialize MPI matching data structures
      rq_max.resize(p);
      uq_max.resize(p); 
  
      for( int i : rq_max ) {
        rq_max[i] = 0;
        uq_max[i] = 0;
      }
  
      rq_matches.resize(p);
      uq_matches.resize(p);
  
      rq_misses.resize(p);
      uq_misses.resize(p);
  
      rq_times.resize(p);
      uq_times.resize(p);
    }
  
    // the active queue 
    std::priority_queue<graph_node_properties,std::vector<graph_node_properties>,aqcompare_func> aq;
    // the queues for each host 
    std::vector<ruq_t> rq(p), uq(p); // receive queue, unexpected queue
    // next available time for o, g(receive) and g(send)
    std::vector<std::vector<btime_t> > nexto(p), nextgr(p), nextgs(p);
  #ifdef HOSTSYNC
    std::vector<btime_t> hostsync(p);
  #endif
  
    // initialize o and g for all PEs and hosts
    for(uint i=0; i<p; ++i) {
      nexto[i].resize(ncpus);
      std::fill(nexto[i].begin(), nexto[i].end(), 0);
      nextgr[i].resize(nnics);
      std::fill(nextgr[i].begin(), nextgr[i].end(), 0);
      nextgs[i].resize(nnics);
      std::fill(nextgs[i].begin(), nextgs[i].end(), 0);
    }
  
      struct timeval tstart, tend;
      gettimeofday(&tstart, NULL);
  
    int host=0; 
    uint64_t num_events=0;
    graph_node_properties recev_msg;
    recev_msg.updated = false;
    bool unlocked_elem = false;

    printf("Starting %lu\n", parser.schedules.size());
    
    for(Parser::schedules_t::iterator sched=parser.schedules.begin(); sched!=parser.schedules.end(); ++sched, ++host) {
      // initialize free operations (only once per run!)
      //sched->init_free_operations();
  
      // retrieve all free operations
      //typedef std::vector<std::string> freeops_t;
      //freeops_t free_ops = sched->get_free_operations();
      SerializedGraph::nodelist_t free_ops;
      sched->GetExecutableNodes(&free_ops);
      // ensure that the free ops are ordered by type
      // std::sort(free_ops.begin(), free_ops.end(), gnp_op_comp_func());
  
      num_events += sched->GetNumNodes();
  
      // walk all new free operations and throw them in the queue 
      for(SerializedGraph::nodelist_t::iterator freeop=free_ops.begin(); freeop != free_ops.end(); ++freeop) {
        //if(print) std::cout << *freeop << " " ;
  
        freeop->host = host;
        freeop->time = 0;
  #ifdef STRICT_ORDER
        freeop->ts=aqtime++;
  #endif
  
        switch(freeop->type) {
          case OP_LOCOP:
            if(print) printf("init %i (%i,%i) loclop: %lu\n", host, freeop->proc, freeop->nic, (long unsigned int) freeop->size);
            break;
          case OP_SEND:
            if(print) printf("init %i (%i,%i) send to: %i, tag: %i, size: %lu\n", host, freeop->proc, freeop->nic, freeop->target, freeop->tag, (long unsigned int) freeop->size);
            break;
          case OP_RECV:
            if(print) printf("init %i (%i,%i) recvs from: %i, tag: %i, size: %lu\n", host, freeop->proc, freeop->nic, freeop->target, freeop->tag, (long unsigned int) freeop->size);
            break;
          default:
            printf("not implemented!\n");
        }
  
        aq.push(*freeop);
        //std::cout << "AQ size after push: " << aq.size() << "\n";
      }
    }

    bool qstat_given = false;
    bool comm_dep_file_given = false;
    bool qstat_arg = false;
    bool progress_given = false;
    bool batchmode_given = false;

    bool new_events=true;
    uint lastperc=0;
    while(!aq.empty() || new_events || (size_queue(rq, p) > 0) || (size_queue(uq, p) > 0)) {
      std::unordered_set<int> check_hosts;
        // get the next element from the queue
        graph_node_properties elem = aq.top();

        while (!aq.empty() && aq.top().time <= (lgs_interface->htsim_time)) {


            graph_node_properties elem = aq.top();
            if (elem.offset % 1000 == 0) {
                printf("Considering Element Host %d Offset %d\n", elem.host, elem.offset);
            }
            aq.pop();
            //std::cout << "AQ size after pop: " << aq.size() << "\n";
            
            // the lists of hosts that actually finished someting -- a host is
            // added whenever an irequires or requires is satisfied
            // print = 1;
            /*if(elem.host == 0) print = 1;
            else print = 0;*/
    
            // the BIG switch on element type that we just found 
            switch(elem.type) {
            case OP_LOCOP: {
                if(print) printf("[%i] found loclop of length %lu - t: %lu (CPU: %i)\n", elem.host, (ulint)elem.size, (ulint)elem.time, elem.proc);
                if(nexto[elem.host][elem.proc] <= elem.time)
                { // local o available!
                    // check if OS Noise occurred
                    uint64_t cpu_time = lgs_interface->htsim_time + elem.size;
                    nexto[elem.host][elem.proc] = cpu_time;

                    if (elem.size == 0) { // TO DO
                        elem.size = 1;
                    }
        
                    if (print)
                        printf("-- nexto[%i][%i] = %lu\n", elem.host, elem.proc, nexto[elem.host][elem.proc]);
                    // satisfy irequires
                    //parser.schedules[elem.host].issue_node(elem.node);
                    // satisfy requires+irequires
                    //parser.schedules[elem.host].remove_node(elem.node);
                    parser.schedules[elem.host].MarkNodeAsStarted(elem.offset);
                    //parser.schedules[elem.host].MarkNodeAsDone(elem.offset, cpu_time);
                    // check_hosts.push_back(elem.host);
                    elem.type = OP_LOCOP_IN_PROGRESS;
                    lgs_interface->compute_started++;
                    check_hosts.insert(elem.host);

                    aq.push(elem);
                    lgs_interface->execute_compute(elem, p);
                    // add to timeline
                    //tlviz.add_loclop(elem.host, elem.time, elem.time+elem.size, elem.proc);
                } else {
                    elem.time = nexto[elem.host][elem.proc];
                    aq.push(std::move(elem)); // TO DO
                    num_reinserts_o++;
                    if(print) printf("-- locop local o not available -- reinserting with t: %lu\n", (long unsigned int) elem.time);
                } 
            } break;
    
            case OP_LOCOP_IN_PROGRESS: {
                parser.schedules[elem.host].MarkNodeAsDone(elem.offset);
            } break;
            
            case OP_SEND: { // a send op
                if(print) printf("[%i] found send to %i tag %lu - t: %lu (CPU: %i)\n", elem.host, elem.target, (ulint)elem.tag, (ulint)elem.time, elem.proc);
    
                uint64_t resource_time = std::max(nexto[elem.host][elem.proc], nextgs[elem.host][elem.nic]);
                if(resource_time <= elem.time) { // local o,g available!
                    if(print) printf("-- satisfy local irequires\n");
                    // check_hosts.push_back(elem.host);
                    check_hosts.insert(elem.host);
                    // FIXME: this is a hack to make sure that the size is at least 1
                    if (elem.size == 0)
                        elem.size = 1;
                    
                    assert(elem.size > 0);
                    // check if OS Noise occurred
                   
                   parser.schedules[elem.host].MarkNodeAsStarted(elem.offset);
                    
                   elem.size *= 1;

                   lgs_interface->htsim_schedule(elem.host, elem.target, elem.size, elem.tag, lgs.htsim_api->getGlobalTimeNs(),
                                                 elem.offset);
        
            #ifdef STRICT_ORDER
                    num_events++; // MSG is a new event
                    elem.ts = aqtime++; // new element in queue 
            #endif    
                } else { // local o,g unavailable - retry later
                if(print) printf("-- send local o,g not available -- reinserting\n");
                    elem.time = resource_time;
                    aq.push(std::move(elem));
                    if (nexto[elem.host][elem.proc] > nextgs[elem.host][elem.nic])
                        num_reinserts_g++;
                    else
                        num_reinserts_g++;
                }
                                                
            } break;
            case OP_RECV: {
                if(print)
                    printf("[%i] found recv from %i tag %lu - t: %lu, label: %lu (CPU: %i)\n",
                            elem.host, elem.target, (ulint)elem.tag, (ulint)elem.time, (ulint)elem.offset + 1, elem.proc);
    
                parser.schedules[elem.host].MarkNodeAsStarted(elem.offset);
                // check_hosts.push_back(elem.host);
                check_hosts.insert(elem.host);
                if(print) printf("-- satisfy local irequires\n");
                
                ruqelem_t matched_elem; 
                // NUMBER of elements that were searched during message matching
                int32_t match_attempts;
    
                if (elem.size == 0)
                    elem.size = 1;
                
                assert(elem.size > 0);
    
                // UPDATE the maximum UQ size
                if( qstat_given ){
                    uq_max[elem.host] = std::max((int)uq[elem.host].size(), uq_max[elem.host]);
                }
                match_attempts = match(elem, &uq[elem.host], &matched_elem);
                if(match_attempts >= 0)
                { // found it in local UQ 
                    // std::cout << "[INFO] Rank " << matched_elem.src << ": " <<
                    //   matched_elem.offset << " -> " << "Rank " << elem.host << ": " << elem.offset << std::endl;
                    if (comm_dep_file_given)
                    {
                        // Stores communication dependencies
                        auto dep = std::make_tuple(matched_elem.src, matched_elem.offset,
                                                elem.host, elem.offset);
                        comm_deps.push_back(dep);
                    }
                    if(print) printf("-- found in local UQ\n");
                    if(qstat_given) {
                        // RECORD match queue statistics
                        std::pair<int,btime_t> match = std::make_pair(match_attempts, elem.time);
                        uq_matches[elem.host].push_back(match);
                        uq_times[elem.host].push_back(elem.time - matched_elem.starttime);
                    }
                    
                    // If the message is found in the unexpected queue, the it should already have been executed
                    assert(elem.time >= matched_elem.starttime);
                    uint64_t nic_time = std::max(nextgs[elem.host][elem.nic], elem.time) + g;
                    uint64_t cpu_time = nic_time + o + (elem.size - 1) * O;
        
                    nexto[elem.host][elem.proc] = cpu_time;
                    nextgr[elem.host][elem.nic] = nic_time;
        
        
                    // satisfy local requires
                    parser.schedules[elem.host].MarkNodeAsDone(elem.offset, cpu_time);
                    // check_hosts.push_back(elem.host);
                    check_hosts.insert(elem.host);
                    if(print) printf("-- satisfy local requires\n");
                } else { // not found in local UQ - add to RQ
                    if(qstat_given) {
                        // RECORD match queue statistics
                        std::pair<int,btime_t> match = std::make_pair((int)uq[elem.host].size(), elem.time);
                        uq_misses[elem.host].push_back(match);
                    }
                    if(print) printf("-- not found in local UQ -- add to RQ\n");
                    ruqelem_t nelem; 
                    nelem.size = elem.size;
                    nelem.src = elem.target;
                    nelem.tag = elem.tag;
                    nelem.offset = elem.offset;
                    //nelem.proc = elem.proc; TO DO
            #ifdef LIST_MATCH
                    rq[elem.host].push_back(nelem);
                    if (print)
                        std::cout << "-- Pushed to RQ: " << nelem.src << " " << nelem.tag << " [RQ size:" << rq[elem.host].size() << "]" << std::endl;
            #else
                    rq[elem.host][std::make_pair(nelem.tag,nelem.src)].push(nelem);
            #endif
                }
            } break;
    
            case OP_MSG: {
                if(print)
                printf("[%i] found msg from %i, t: %lu tag: %i, (CPU: %i)\n", elem.host, elem.target, (ulint)elem.time, elem.tag, elem.proc);
                uint64_t earliestfinish;
                // NUMBER of elements that were searched during message matching
                int32_t match_attempts;
    
                // if(true || (earliestfinish = net.query(elem.starttime, elem.time, elem.target, elem.host, elem.size, &elem.handle)) <= elem.time)
                /* msg made it through network */ // &&
                //    std::max(nexto[elem.host][elem.proc],nextgr[elem.host][elem.nic]) <= elem.time /* local o,g available! */) { 
                //   if(print)
                //     printf("-- msg o,g available (nexto: %lu, nextgr: %lu)\n", (long unsigned int) nexto[elem.host][elem.proc], (long unsigned int) nextgr[elem.host][elem.nic]);          
                ruqelem_t matched_elem;
                match_attempts = match(elem, &rq[elem.host], &matched_elem);
                if(match_attempts >= 0)
                { // found it in RQ
    
                    if (elem.size == 0)
                    elem.size = 1;
                    
                    assert(elem.size > 0);    
                    
                    // std::cout << "[INFO] Rank " << matched_elem.src << ": " <<
                    // matched_elem.offset << " -> " << "Rank " << elem.host << ": " << elem.offset << std::endl;
                    if (comm_dep_file_given)
                    {
                        // Stores communication dependencies
                        auto dep = std::make_tuple(elem.target, elem.offset,
                                                    elem.host, matched_elem.offset);
                        comm_deps.push_back(dep);
                    }
    
                    if(qstat_given) {
                        // RECORD match queue statistics
                        std::pair<int,btime_t> match = std::make_pair(match_attempts, elem.time);
                        rq_matches[elem.host].push_back(match);
                        /* Amount of time spent in queue */
                        rq_times[elem.host].push_back(elem.time - matched_elem.starttime);
                    }
    
                    if(print)
                        printf("-- found in RQ\n");
                    
                    parser.schedules[elem.host].MarkNodeAsDone(matched_elem.offset);
                    parser.schedules[elem.target].MarkNodeAsDone(elem.offset);
                    /* tlviz.add_transmission(elem.target, elem.host, elem.starttime+o, elem.time, elem.size, G);
                    tlviz.add_orecv(elem.host, elem.time+ static_cast<uint64_t>((elem.size-1)*G)-(elem.size-1)*O,
                                    elem.time+o+std::max((elem.size-1)*O, static_cast<uint64_t>((elem.size-1)*G)), elem.proc);
                    tlviz.add_noise(elem.host,
                                    elem.time + o + std::max((elem.size-1) * O, static_cast<uint64_t>((elem.size - 1) * G)),
                                    elem.time + o + noise + std::max((elem.size-1)*O, static_cast<uint64_t>((elem.size - 1) * G)), elem.proc); */
                    // satisfy local requires
    
                    // check_hosts.push_back(elem.host);
                    check_hosts.insert(elem.host);
                    if (print)
                        printf("-- satisfy local requires\n"); 
                }
                else
                { // not in RQ
                    if(qstat_given) {
                    // RECORD match queue statistics
                    std::pair<int,btime_t> match = std::make_pair((int)rq[elem.host].size(), elem.time);
                    rq_misses[elem.host].push_back(match);
                    }
    
                    if(print) printf("-- not found in RQ - add to UQ\n");
                    ruqelem_t nelem;
                    nelem.size = elem.size;
                    nelem.src = elem.target;
                    nelem.tag = elem.tag;
                    nelem.offset = elem.offset;
                    nelem.starttime = elem.time; // when it was started
        #ifdef LIST_MATCH
                    uq[elem.host].push_back(nelem);
        #else
                    uq[elem.host][std::make_pair(nelem.tag, nelem.src)].push(nelem);
        #endif
                }                
            } break;
                
            default:
                printf("not supported\n");
                break;
            
        }
      }
  
      // do only ask hosts that actually completed something in this round!
      new_events = false;
      // std::sort(check_hosts.begin(), check_hosts.end());
      // check_hosts.erase(unique(check_hosts.begin(), check_hosts.end()), check_hosts.end());
      


      if (print)
        std::cout << "[INFO] Checking for free operations on hosts:" << std::endl;
      // for(std::vector<int>::iterator iter=check_hosts.begin(); iter!=check_hosts.end(); ++iter) {
      for (int host : check_hosts) {
        // host = *iter;
      //for(host = 0; host < p; host++) 
        SerializedGraph *sched=&parser.schedules[host];
  
        // retrieve all free operations
        SerializedGraph::nodelist_t free_ops;
        sched->GetExecutableNodes(&free_ops);
        // ensure that the free ops are ordered by type
        // std::sort(free_ops.begin(), free_ops.end(), gnp_op_comp_func());
        
        // walk all new free operations and throw them in the queue 
        for(SerializedGraph::nodelist_t::iterator freeop=free_ops.begin(); freeop != free_ops.end(); ++freeop) {
          //if(print) std::cout << *freeop << " " ;
  
          new_events = true;
  
          // assign host that it starts on
          freeop->host = host;
  
  #ifdef STRICT_ORDER
          freeop->ts=aqtime++;
  #endif
          
          switch(freeop->type) {
            case OP_LOCOP:
              // freeop->time = nexto[host][freeop->proc];
              freeop->time = std::max(freeop->starttime, nexto[host][freeop->proc]);
              if(print)
                printf("Freeop %i (%i,%i) loclop: %lu, time: %lu, offset: %i\n", host, freeop->proc, freeop->nic, (long unsigned int) freeop->size, (long unsigned int)freeop->time, freeop->offset);
              break;
            case OP_SEND:
              // freeop->time = std::max(nexto[host][freeop->proc], nextgs[host][freeop->nic]);
              freeop->time = std::max(freeop->starttime, nextgs[host][freeop->nic]);
              // if (freeop->proc == 73)
              // {
              //   std::cout << "[DEBUG] HOST " << host << " Send offset " << freeop->offset << " NextO: " << nexto[host][freeop->proc] / 1e9 << " Time: " << freeop->time / 1e9 << std::endl;
              // }
              if(print)
                printf("Freeop %i (%i,%i) send to: %i, tag: %i, size: %lu, time: %lu, offset: %i\n", host, freeop->proc, freeop->nic, freeop->target, freeop->tag, (long unsigned int) freeop->size, (long unsigned int)freeop->time, freeop->offset);
              break;
            case OP_RECV:
              // freeop->time = nexto[host][freeop->proc];
              freeop->time = freeop->starttime;
              if(print)
                printf("Freeop %i (%i,%i) recvs from: %i, tag: %i, size: %lu, time: %lu, offset: %i\n", host, freeop->proc, freeop->nic, freeop->target, freeop->tag, (long unsigned int) freeop->size, (long unsigned int)freeop->time, freeop->offset);
              break;
            default:
              printf("not implemented!\n");
          }
          freeop->time = lgs.htsim_api->getGlobalTimeNs();
          aq.push(*freeop);
        }
      }


      if (!lgs_interface->all_sends_delivered() || lgs_interface->compute_started != 0) {
            if (!unlocked_elem) {
                recev_msg = lgs_interface->htsim_simulate_until(100);
                lgs_interface->htsim_time = lgs.htsim_api->getGlobalTimeNs();
            }
        }

        // If the OP is NULL then we just continue
        if (recev_msg.updated) {
            aq.push(recev_msg);
            lgs_interface->reset_latest_receive();
        } else {
        }

      if (print)
        std::cout << "[INFO] Finished checking for free operations on hosts. [AQ size: " << aq.size() << "]" << std::endl;
      if(progress_given) {
        if(num_events/100*lastperc < aqtime) {
          printf("progress: %u %% (%llu) ", lastperc, (unsigned long long)aqtime);
          lastperc++;
          uint maxrq=0, maxuq=0;
          for(uint j=0; j<rq.size(); j++) maxrq=std::max(maxrq, (uint)rq[j].size());
          for(uint j=0; j<uq.size(); j++) maxuq=std::max(maxuq, (uint)uq[j].size());
          printf("[sizes: aq: %i max rq: %u max uq: %u: reinserts: (%lu, %lu, %lu)]\n", (int)aq.size(), maxrq, maxuq, num_reinserts_o, num_reinserts_g, num_reinserts_net);
          num_reinserts_o=0;
          num_reinserts_g=0;
          num_reinserts_net=0;
        }
      }
    }
      
    gettimeofday(&tend, NULL);
      unsigned long int diff = tend.tv_sec - tstart.tv_sec;
  
  #ifndef STRICT_ORDER
    ulint aqtime=0;
  #endif
      printf("PERFORMANCE: Processes: %i \t Events: %lu \t Time: %lu s \t Speed: %.2f ev/s\n", p, (long unsigned int)aqtime, (long unsigned int)diff, (float)aqtime/(float)diff);
  
    // check if all queues are empty!!
    bool ok=true;
    for(uint i=0; i<p; ++i) {
      
  #ifdef LIST_MATCH
      if(!uq[i].empty()) {
        printf("unexpected queue on host %i contains %lu elements!\n", i, (ulint)uq[i].size());
        for(ruq_t::iterator iter=uq[i].begin(); iter != uq[i].end(); ++iter) {
          printf(" src: %i, tag: %u, size: %u\n", iter->src, iter->tag, iter->size);
        }
        ok=false;
      }
      if(!rq[i].empty()) {
        printf("receive queue on host %i contains %lu elements!\n", i, (ulint)rq[i].size());
        for(ruq_t::iterator iter=rq[i].begin(); iter != rq[i].end(); ++iter) {
          printf(" src: %i, tag: %u, offset: %u, size: %u\n", iter->src, iter->tag, iter->offset, iter->size);
        }
        ok=false;
      }
  #endif
  
    }
    
    if (ok) {
      if(p <= 16 && !batchmode_given) { // print all hosts
        printf("Times: \n");
        host = 0;
        for(uint i=0; i<p; ++i) {
          btime_t maxo=*(std::max_element(nexto[i].begin(), nexto[i].end()));
          //btime_t maxgr=*(std::max_element(nextgr[i].begin(), nextgr[i].end()));
          //btime_t maxgs=*(std::max_element(nextgs[i].begin(), nextgs[i].end()));
          //std::cout << "Host " << i <<": "<< std::max(std::max(maxgr,maxgs),maxo) << "\n";
          std::cout << "Host " << i <<": "<< maxo << "\n";
          // if (i == 0)
          // {
          //   for (int cpu = 0; cpu < ncpus; ++cpu)
          //   {
          //     std::cout << "NextO[" << i << "][" << cpu << "]: " << nexto[i][cpu] << std::endl;
          //   }
          // }
        }
      } else { // print only maximum
        long long unsigned int max=0;
        int host=0;
        for(uint i=0; i<p; ++i) { // find maximum end time
          btime_t maxo=*(std::max_element(nexto[i].begin(), nexto[i].end()));
          //btime_t maxgr=*(std::max_element(nextgr[i].begin(), nextgr[i].end()));
          //btime_t maxgs=*(std::max_element(nextgs[i].begin(), nextgs[i].end()));
          //btime_t cur = std::max(std::max(maxgr,maxgs),maxo);
          btime_t cur = maxo;
          if(cur > max) {
            host=i;
            max=cur;
          }
        }
        std::cout << "Maximum finishing time at host " << host << ": " << max << " ("<<(double)max/1e9<< " s)\n";
      }
      // Outputs recorded communication dependencies
      if (comm_dep_file_given)
      {
        std::ofstream comm_dep_file("test");
        if ( !comm_dep_file.is_open() )
        {
          std::cerr << "[ERROR] Cannot open file: " << comm_dep_file_arg << std::endl;
        }
        else
        {
          for (const auto& dep : comm_deps)
          {
            comm_dep_file << std::get<0>(dep) << "," << std::get<1>(dep) << "," <<
                             std::get<2>(dep) << "," << std::get<3>(dep) << "\n";
          }
          comm_dep_file.close();
          std::cout << "[INFO] Comm dep file saved to " << comm_dep_file_arg << std::endl;
        }
      }
  
      // WRITE match queue statistics
      if( qstat_given ){
        char filename[1024];
  
        // Maximum RQ depth
        snprintf(filename, sizeof filename, "%s-%s", qstat_arg, "rq-max.data");
        std::ofstream rq_max_file(filename);
  
        if( !rq_max_file.is_open() ) {
          std::cerr << "Can't open rq-max data file" << std::endl;
        } else {
          // WRITE one line per rank
          for( int n : rq_max ) {
            rq_max_file << n << std::endl;
          }
        }
        rq_max_file.close();
  
        // Maximum UQ depth
        snprintf(filename, sizeof filename, "%s-%s", qstat_arg, "uq-max.data");
        std::ofstream uq_max_file(filename);
  
        if( !uq_max_file.is_open() ) {
          std::cerr << "Can't open uq-max data file" << std::endl;
        } else {
          // WRITE one line per rank
          for( int n : uq_max ) {
            uq_max_file << n << std::endl;
          }
        }
        uq_max_file.close();
  
        // RQ hit depth (number of elements searched for each successful search)
        snprintf(filename, sizeof filename, "%s-%s", qstat_arg, "rq-hit.data");
        std::ofstream rq_hit_file(filename);
  
        if( !rq_hit_file.is_open() ) {
          std::cerr << "Can't open rq-hit data file (" << filename << ")" << std::endl;
        } else {
          // WRITE one line per rank
          for( auto per_rank_matches = rq_matches.begin(); 
                    per_rank_matches != rq_matches.end();  
                    per_rank_matches++ ) 
          {
            for( auto match_pair = (*per_rank_matches).begin(); 
                      match_pair != (*per_rank_matches).end(); 
                      match_pair++ ) 
            {
              rq_hit_file << (*match_pair).first << "," << (*match_pair).second << " ";
            }
            rq_hit_file << std::endl;
          }
        }
        rq_hit_file.close();
  
        // UQ hit depth (number of elements searched for each successful search)
        snprintf(filename, sizeof filename, "%s-%s", qstat_arg, "uq-hit.data");
        std::ofstream uq_hit_file(filename);
  
        if( !uq_hit_file.is_open() ) {
          std::cerr << "Can't open uq-hit data file (" << filename << ")" << std::endl;
        } else {
          // WRITE one line per rank
          for( auto per_rank_matches = uq_matches.begin(); 
                    per_rank_matches != uq_matches.end();  
                    per_rank_matches++ ) 
          {
            for( auto match_pair = (*per_rank_matches).begin(); 
                      match_pair != (*per_rank_matches).end(); 
                      match_pair++ ) 
            {
              uq_hit_file << (*match_pair).first << "," << (*match_pair).second << " ";
            }
            uq_hit_file << std::endl;
          }
        }
        uq_hit_file.close();
  
        // RQ miss depth (number of elements searched for each unsuccessful search)
        snprintf(filename, sizeof filename, "%s-%s", qstat_arg, "rq-miss.data");
        std::ofstream rq_miss_file(filename);
  
        if( !rq_miss_file.is_open() ) {
          std::cerr << "Can't open rq-miss data file (" << filename << ")" << std::endl;
        } else {
          // WRITE one line per rank
          for( auto per_rank_misses = rq_misses.begin(); 
                    per_rank_misses != rq_misses.end();  
                    per_rank_misses++ ) 
          {
            for( auto miss_pair = (*per_rank_misses).begin(); 
                      miss_pair != (*per_rank_misses).end(); 
                      miss_pair++ ) 
            {
              rq_miss_file << (*miss_pair).first << "," << (*miss_pair).second << " ";
            }
            rq_miss_file << std::endl;
          }
        }
        rq_miss_file.close();
  
        // UQ miss depth (number of elements searched for each unsuccessful search)
        snprintf(filename, sizeof filename, "%s-%s", qstat_arg, "uq-miss.data");
        std::ofstream uq_miss_file(filename);
  
        if( !uq_miss_file.is_open() ) {
          std::cerr << "Can't open uq-miss data file (" << filename << ")" << std::endl;
        } else {
          // WRITE one line per rank
          for( auto per_rank_misses = uq_misses.begin(); 
                    per_rank_misses != uq_misses.end();  
                    per_rank_misses++ ) 
          {
            for( auto miss_pair = (*per_rank_misses).begin(); 
                      miss_pair != (*per_rank_misses).end(); 
                      miss_pair++ ) 
            {
              uq_miss_file << (*miss_pair).first << "," << (*miss_pair).second << " ";
            }
            uq_miss_file << std::endl;
          }
        }
        uq_miss_file.close();
      }
    }
  }


int size_queue(std::vector<ruq_t> my_queue, int num_proce) {
    std::size_t max = 0;
    for (int i = 0; i < num_proce; i++) {
        if (my_queue[i].size() > max) {
            max = my_queue[i].size();
        }
    }
    return max;
}
