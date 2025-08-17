/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "logsim-interface.h"
#include "lgs/LogGOPSim.hpp"
//#include "lgs/Network.hpp"
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
#ifndef LIST_MATCH
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif
#endif

#define DEBUG_PRINT 0

#ifndef LIST_MATCH
namespace std
{
   using namespace __gnu_cxx;
}
#endif

static bool print = false;


LogSimInterface::LogSimInterface() {}

LogSimInterface::LogSimInterface(EqdsLogger *logger, TrafficLoggerSimple *pktLogger, EventList &eventList,
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
    if (null_events_handler == NULL) {
      null_events_handler = new NullEvent(*_eventlist);
      null_events_handler->set_null_over_hook(
              std::bind(&LogSimInterface::null_over, this, std::placeholders::_1));
  }
}

LogSimInterface::LogSimInterface(EqdsLogger *logger, NdpTrafficLogger pktLogger, EventList &eventList,
  FatTreeTopology *topo, std::vector<const Route *> ***routes) {
  _logger = logger;
  _eventlist = &eventList;
  _topo = topo;
  _netPaths = routes;
  _latest_recv = new graph_node_properties();
  if (compute_events_handler == NULL) {
  compute_events_handler = new ComputeEvent(*_eventlist);
  compute_events_handler->set_compute_over_hook(
  std::bind(&LogSimInterface::compute_over, this, std::placeholders::_1));
  }
  if (null_events_handler == NULL) {
  null_events_handler = new NullEvent(*_eventlist);
  null_events_handler->set_null_over_hook(
  std::bind(&LogSimInterface::null_over, this, std::placeholders::_1));
  }
}

void LogSimInterface::set_cwd(int cwd) { _cwd = cwd; }

void LogSimInterface::htsim_schedule(u_int32_t host, int to, int size, int tag, u_int64_t start_time_event,
                                     int my_offset) {
    // Send event to htsim for actual send
    //send_event(host, to, size, tag, start_time_event);

    // Save Event for internal tracking
    std::string to_hash = std::to_string(host) + "@" + std::to_string(to) + "@" + std::to_string(tag);
    /* printf("Scheduling Event (%s) of size %d from %d to %d tag %d start_tiem "
           "%lu - Time is %lu\n ",
           to_hash.c_str(), size, host, to, tag, start_time_event * 1000, GLOBAL_TIME); */
    /* MsgInfo entry;
    entry.start_time = start_time_event * 1;
    entry.total_bytes_msg = size;
    entry.offset = my_offset;
    entry.bytes_left_to_recv = size;
    entry.to_parse = 42;
    active_sends[to_hash] = entry; */
}

void LogSimInterface::execute_compute(graph_node_properties comp_elem, int size_p) {
    if (_protocolName == EQDS_PROTOCOL || _protocolName == NDP_PROTOCOL || _protocolName == SENDER_PROTOCOL) {
        compute_events_handler->setCompute(comp_elem.size * 1);
        ComputeAtlahsEvent *compute_event = new ComputeAtlahsEvent(comp_elem.size);
        htsim_api->Calc(*compute_event);
    }
}

void LogSimInterface::execute_null_compute(graph_node_properties comp_elem, int size_p) {
  if (_protocolName == EQDS_PROTOCOL || _protocolName == NDP_PROTOCOL || _protocolName == SENDER_PROTOCOL) {
      
      null_events_handler->setCompute(comp_elem.size);
  }
}

void LogSimInterface::send_event(graph_node_properties elem) {

    std::string to_hash = std::to_string(elem.host) + "@" + std::to_string(elem.target) + "@" + std::to_string(elem.tag);
    /* printf("Scheduling Event (%s) of size %d from %d to %d tag %d start_tiem "
           "%lu - Time is %lu\n ",
           to_hash.c_str(), size, host, to, tag, start_time_event * 1000, GLOBAL_TIME); */
    /* MsgInfo entry;
    entry.start_time = htsim_api->getGlobalTimeNs();
    entry.total_bytes_msg = elem.size;
    entry.offset = elem.offset;
    entry.bytes_left_to_recv = elem.size;
    entry.to_parse = 42;
    active_sends[to_hash] = entry; */

    // Testing New 
    SendEvent* event = new SendEvent(elem.host, elem.target, elem.size, elem.tag, htsim_api->getGlobalTimeNs());    
    htsim_api->Send(*event, elem);

    /* printf("LGS Send Event - Time %lu - Host %d - Dst %d - Tag %d - Size %d - "
           "StartTime %d\n",
           1000 / 1000, from, to, tag, size, start_time_event);  */

    return;
}

void LogSimInterface::update_active_map(std::string to_hash, int size) {

    // Check that the flow actually exists
    active_sends[to_hash].bytes_left_to_recv = active_sends[to_hash].bytes_left_to_recv - Packet::data_packet_size();
    if (active_sends[to_hash].bytes_left_to_recv <= 0) {
    }
}

bool LogSimInterface::all_sends_delivered() { return active_sends.size() == 0; }

void LogSimInterface::flow_over(const EventOver &event) {
  sends_active--;
  debug_stop--;

  // active_sends[to_hash].bytes_left_to_recv = 0;
  // Here we have received a message fully, we need to give control back to
  // LGS
  _latest_recv = new graph_node_properties();
  _latest_recv->updated = true;
  _latest_recv->tag = event.node->tag;
  _latest_recv->type = OP_MSG;
  _latest_recv->target = event.node->host;
  _latest_recv->host = event.node->target;
  _latest_recv->starttime = event.start_time_event;
  _latest_recv->time = htsim_api->getGlobalTimeNs();
  _latest_recv->size = event.node->size;
  _latest_recv->offset = event.node->offset;
  _latest_recv->proc = event.node->proc;
  _latest_recv->nic = event.node->nic;

  aq.push(*_latest_recv);
}

void LogSimInterface::compute_over(int i) {
    compute_started--;
    compute_if_finished = true;
    return;
}

void LogSimInterface::null_over(int i) {
  compute_started--;
  htsim_api->send_done_return_control = true;
  return;
}

void LogSimInterface::reset_latest_receive() { _latest_recv->updated = false; }

void LogSimInterface::terminate_sim() {
  uint64_t tot_nacks = 0;
  if (print_stats_flows) {
    printf("Flow Info Size is %lu\n", htsim_api->flowInfos.size());
    for (int i = 0; i < htsim_api->flowInfos.size(); i++) {
      printf("F %d - ST %lu - ET %lu - OT %lu - S %lu - N %lu - C %lu\n",
             i, htsim_api->flowInfos[i].flowStartTime, htsim_api->flowInfos[i].flowEndTime,
             htsim_api->flowInfos[i].completionTime, htsim_api->flowInfos[i].flowSize,
             htsim_api->flowInfos[i].numNacks, htsim_api->flowInfos[i].finalCwnd);
      tot_nacks += htsim_api->flowInfos[i].numNacks;

    }
    printf("\nTotal Number of NACKS %lu\n", tot_nacks);
  }
}

void LogSimInterface::htsim_simulate_until(int64_t until) {

    //printf("Running HTSIM Simulate Until %lu - HtSime Time is %lu\n", until, htsim_api->getGlobalTimeNs());

    if (until != -1) {
      compute_started++;
      null_events_handler->setCompute(until);
    }

    have_more = false;

    //printf("Size eventlist is %lu\n", _eventlist->getPendingSources().size());

    while (_eventlist->doNextEvent()) {

        if (htsim_api->send_done_return_control) {
            htsim_api->send_done_return_control = false;

            if (_eventlist->now() == _eventlist->getPendingSources().begin()->first) {
              have_more = true;
            }
            break;
        }

        //printf("While3 - Time is %lu - Have More %d\n", _eventlist->now(), have_more);
        if (_latest_recv->updated) {
          this->reset_latest_receive();
          if (_eventlist->now() == _eventlist->getPendingSources().begin()->first) {
            have_more = true;
          }
            break;
        }

        if (compute_if_finished) {
          if (_eventlist->now() == _eventlist->getPendingSources().begin()->first) {
            have_more = true;
          }
            compute_if_finished = false;
            break;
        }
    }

    return;
}

typedef struct {
  // TODO: src and tag can go in case of hashmap matching
  btime_t starttime; // only for visualization
  uint32_t size, src, tag, offset, proc;
} ruqelem_t;

typedef unsigned int uint;
typedef unsigned long int ulint;
typedef unsigned long long int ullint;

#ifdef LIST_MATCH
// TODO this is really slow - reconsider design of rq and uq!
// matches and removes element from list if found, otherwise returns
// false
typedef std::list<ruqelem_t> ruq_t;
static inline int match(const graph_node_properties &elem, ruq_t *q, ruqelem_t *retelem=NULL) {

  // MATCH attempts (i.e., number of elements searched to find a matching element)
  int match_attempts = 0;
  //std::cout << "UQ size " << q->size() << "\n";

  if(print)
    printf("++ [%i] searching matching queue for src %i tag %i\n", elem.host, elem.target, elem.tag);
  // Iterate from the end of the queue
  int curr_offset = -1;
  // Only returns the matched element with the smallest offset
  ruq_t::iterator matched_iter = q->end();
  for(ruq_t::iterator iter=q->begin(); iter!=q->end(); ++iter) {
    match_attempts++;
    if(elem.target == ANY_SOURCE || iter->src == ANY_SOURCE || iter->src == elem.target) {
      if(elem.tag == ANY_TAG || iter->tag == ANY_TAG || iter->tag == elem.tag) {
        if (curr_offset == -1)
        {
          *retelem = *iter;
          matched_iter = iter;
          curr_offset = iter->offset;
        }
        else if (iter->offset < curr_offset)
        {
          *retelem = *iter;
          matched_iter = iter;
        }
        // if(retelem)
        //   *retelem=*iter;
        // q->erase(iter);
        // return match_attempts;
      }
    }
  }
  if (curr_offset != -1)
  {
    q->erase(matched_iter);
    return match_attempts;
  }
  return -1;
}
#else
class myhash { // I WANT LAMBDAS! :)
  public:
  size_t operator()(const std::pair<int,int>& x) const {
    return (x.first>>16)+x.second;
  }
};
typedef std::hash_map< std::pair</*tag*/int,int/*src*/>, std::queue<ruqelem_t>, myhash > ruq_t;
static inline int match(const graph_node_properties &elem, ruq_t *q, ruqelem_t *retelem=NULL) {
  
  if(0) printf("++ [%i] searching matching queue for src %i tag %i\n", elem.host, elem.target, elem.tag);

  ruq_t::iterator iter = q->find(std::make_pair(elem.tag, elem.target));
  if(iter == q->end()) {
    return -1;
  }
  std::queue<ruqelem_t> *tq=&iter->second;
  if(tq->empty()) {
    return -1;
  }
  if(retelem) *retelem=tq->front();
  tq->pop();
  //q->erase(iter);
  return 0;
}
#endif

int size_queue(std::vector<ruq_t> my_queue, int num_proce);


int start_lgs(std::string filename_goal, LogSimInterface &lgs) {
    LogSimInterface *lgs_interface = &lgs;


    #ifdef STRICT_ORDER
    btime_t aqtime=0; 
    uint64_t num_reinserts_o = 0;
    uint64_t num_reinserts_g = 0;
    uint64_t num_reinserts_net = 0;
  #endif

  #ifndef LIST_MATCH
    if( false ){  // TO DO
      printf("WARNING: --qstat option provided, but LogGOPSim was compiled with LIST_MATCH;\n"
             "         statistics on match queue behavior are NOT valid.\n"); 
    }
  #endif
  
    // read input parameters
    const int o = 0;
    const int O = 0;
    const int g = 0;
    const int L = 0;
    const double G = 0.04;
    print = false;
    const uint32_t S = 0;
    bool custom_print = false;

    for (int i = 0; i < 8192; ++i)
    {
        lgs_interface->nic_available[i] = true;
    }

    printf("filename is %s\n", filename_goal.c_str());

    Parser parser(filename_goal, false);
  
    const uint p = parser.schedules.size();
    const int ncpus = parser.GetNumCPU();
    const int nnics = parser.GetNumNIC();
    bool comm_dep_file_arg = false;

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
    //std::priority_queue<graph_node_properties,std::vector<graph_node_properties>,aqcompare_func> aq;
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
            if(0) printf("init %i (%i,%i) loclop: %lu\n", host, freeop->proc, freeop->nic, (long unsigned int) freeop->size);
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
  
        lgs_interface->aq.push(*freeop);
      }
    }

    bool qstat_given = false;
    bool comm_dep_file_given = false;
    bool qstat_arg = false; 
    bool batchmode_given = false;
    lgs_interface->htsim_api->setNumberNic(nnics);
    bool progress_given = true;

    
    int count_cycless = 0;
    bool new_events=true;
    uint lastperc=0;
    int send_doing = 0;

    while(!lgs_interface->aq.empty() || new_events || lgs_interface->sends_active > 0 ||  lgs_interface->compute_started > 0) {

      count_cycless++;
      if (count_cycless > 20000000) {
          //printf("Count1 Cycles Exceeded 200000\n");
          //exit(0);
      }
      int count_cycles = 0;
      int64_t can_simulate_until = -1;
      
      /* if (count_cycless % 1000000 == 0) {
        printf("------ HtSim Time %lu --------------------    ENTERING WHILE        // "
          "---------------------------- | %d - Top %ld vs Htsim %ld -  %d %d - %d %d "
          "Size AQ %lu - "
          "host %d target %d  tag %d type %d  -- S active %d C active %d\n", lgs_interface->htsim_api->getGlobalTimeNs(),
          lgs_interface->aq.empty(), lgs_interface->aq.top().time, lgs_interface->htsim_api->getGlobalTimeNs(), 99, 99, size_queue(rq, p),
          size_queue(uq, p), lgs_interface->aq.size(), lgs_interface->aq.top().host, lgs_interface->aq.top().target, lgs_interface->aq.top().tag, lgs_interface->aq.top().type,
          lgs_interface->sends_active, lgs_interface->compute_started); 
      } */         
      
      std::unordered_set<int> check_hosts;
        // get the next element from the queue
        graph_node_properties elem = lgs_interface->aq.top();

        while (!lgs_interface->aq.empty() && !lgs_interface->have_more && lgs_interface->aq.top().time <= (lgs_interface->htsim_api->getGlobalTimeNs())) {   
          /* printf("Active Queue Size %d - Type %d - Host %d - CPU %d - Tag %d - Top Time %lu -- Size RQ %d -- Size UQ %d\n",
            (int)lgs_interface->aq.size(), lgs_interface->aq.top().type, lgs_interface->aq.top().host, lgs_interface->aq.top().proc, lgs_interface->aq.top().tag, lgs_interface->aq.top().time, 
            size_queue(rq, p), size_queue(uq, p));  */

          /* printf("Active Queue Size %d - Top Time %lu Type %d - HTSIM Time "
            "%lu - Host %d - Proc %d - NIC Busy %d\n",
            (int)lgs_interface->aq.size(), lgs_interface->aq.top().time, lgs_interface->aq.top().type, lgs_interface->htsim_api->getGlobalTimeNs(), lgs_interface->aq.top().host, lgs_interface->aq.top().proc, lgs_interface->nic_available[lgs_interface->aq.top().host]);  */

            count_cycles++;
            if (count_cycles > 200000000) {
                //printf("Count Cycles Exceeded 200000\n");
                //exit(0);
            }
            graph_node_properties elem = lgs_interface->aq.top();

            if (elem.offset % 100 == 0) {
                //printf("Considering Element Host %d Offset %d\n", elem.host, elem.offset);
            }
            lgs_interface->aq.pop();

            // Update time
            //elem.nic = 0;
            if (elem.time < lgs_interface->htsim_api->getGlobalTimeNs()) {
              elem.time = lgs_interface->htsim_api->getGlobalTimeNs();
            }

            if (elem.host == elem.target && (elem.type == OP_SEND || elem.type == OP_RECV)) {
              elem.type = OP_LOCOP;
              elem.size = 1;
              lgs_interface->aq.push(elem);
            }

            // the BIG switch on element type that we just found 
            switch(elem.type) {
            case OP_LOCOP: {
                if(0) printf("[%i] found loclop of length %lu - offset: %d -  t: %lu (CPU: %i)\n", elem.host, (ulint)elem.size, elem.offset, (ulint)elem.time, elem.proc);
                if(nexto[elem.host][elem.proc] <= elem.time) { // local o available!

                    if (elem.size == 0) { // Hack for 0 Computes
                      elem.size = 1;
                    }

                    // Update CPU time
                    btime_t noise = 0; // No noise with HTSIM
                    uint64_t cpu_time = elem.time + elem.size + noise;
                    nexto[elem.host][elem.proc] = cpu_time;
        
                    if (print)
                        printf("-- nexto[%i][%i] = %lu\n", elem.host, elem.proc, nexto[elem.host][elem.proc]);

                    // Mark Compute As Started
                    parser.schedules[elem.host].MarkNodeAsStarted(elem.offset);

                    // Update Element
                    elem.type = OP_LOCOP_IN_PROGRESS;
                    elem.time = cpu_time;
                    lgs_interface->compute_started++;
                    check_hosts.insert(elem.host);

                    // Re-Insert Compute with in progress type and execute it on HTSIM
                    lgs_interface->aq.push(elem);
                    lgs_interface->execute_compute(elem, p);
                    if(0) printf("[%i] done loclop of length %lu - offset: %d -  t: %lu (CPU: %i)\n", elem.host, (ulint)elem.size, elem.offset, (ulint)elem.time, elem.proc);
                } else {
                    elem.time = nexto[elem.host][elem.proc];
                    lgs_interface->aq.push(std::move(elem)); 
                    num_reinserts_o++;
                    if(0) printf("[%i] reinsert loclop of length %lu - offset: %d -  t: %lu (CPU: %i)\n", elem.host, (ulint)elem.size, elem.offset, (ulint)elem.time, elem.proc);
                    if(print) printf("-- locop local o not available -- reinserting with t: %lu\n", (long unsigned int) elem.time);
                } 
            } break;
    
            case OP_LOCOP_IN_PROGRESS: {
              uint64_t cpu_time = elem.time;
              //nexto[elem.host][elem.proc] = cpu_time;  
              parser.schedules[elem.host].MarkNodeAsDone(elem.offset, cpu_time);
              check_hosts.insert(elem.host);
            } break;
            
            case OP_SEND: { // a send op
              if(print) printf("[%i] found send to %i tag %lu - t: %lu (CPU: %i)\n", elem.host, elem.target, (ulint)elem.tag, (ulint)elem.time, elem.proc);
  
              uint64_t resource_time = std::max(nexto[elem.host][elem.proc], nextgs[elem.host][elem.nic]);
              if(nextgs[elem.host][elem.nic] <= elem.time) { // local o,g available!
                  if(print) 
                    printf("-- satisfy local irequires\n");

                  // We Mark the Send as started, before passing it to htsim
                  parser.schedules[elem.host].MarkNodeAsStarted(elem.offset);
                  check_hosts.insert(elem.host);
                  check_hosts.insert(elem.target);

                  // This is a hack to make sure that the size is at least 1
                  if (elem.size == 0)
                      elem.size = 1;
                  assert(elem.size > 0);

                  // Update CPU Availability
                  btime_t noise = 0;
                  uint64_t cpu_time = elem.time + lgs_interface->lgs_o + noise; // We don't model the other paramter in HTSIM
                  nexto[elem.host][elem.proc] = cpu_time;
                  // Calc Actual Size in HTSIM
                  int original_size = elem.size;
                  int message_size = elem.size;
                  int packet_size = 4096;
                  int num_packets = (message_size + packet_size - 1) / packet_size;
                  int updated_size = (num_packets+1) * 4160; // Parameterize this. This accounts for the header size
                  elem.size = updated_size;

                  uint64_t bandwidth_cost2 = static_cast<uint64_t>((elem.size) * G);
                  nextgs[elem.host][elem.nic] = elem.time + g + bandwidth_cost2; 
                  can_simulate_until = nextgs[elem.host][elem.nic];

                  lgs_interface->nic_available[elem.host] = false;
                  lgs_interface->sends_active++;
                  elem.size = original_size;
                  lgs_interface->send_event(elem);
                  send_doing++;

          #ifdef STRICT_ORDER
                  num_events++; // MSG is a new event
                  elem.ts = aqtime++; // new element in queue 
          #endif    
              } else { // local o,g unavailable - retry later
              if(print) printf("-- send local o,g not available -- reinserting\n");
                  //printf("Reinseringg send %d %d %d %d at %lu\n", elem.host, elem.target, elem.tag, elem.size, resource_time);
                  elem.time = resource_time;
                  lgs_interface->aq.push(std::move(elem));
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
                if(match_attempts >= 0) { // found it in local UQ 
                    if (comm_dep_file_given) {
                        // Stores communication dependencies
                        auto dep = std::make_tuple(matched_elem.src, matched_elem.offset,
                                                elem.host, elem.offset);
                        comm_deps.push_back(dep);
                    }
                    if(print) 
                      printf("-- found in local UQ\n");

                    if(qstat_given) {
                        // RECORD match queue statistics
                        std::pair<int,btime_t> match = std::make_pair(match_attempts, elem.time);
                        uq_matches[elem.host].push_back(match);
                        uq_times[elem.host].push_back(elem.time - matched_elem.starttime);
                    }
                    
                    assert(elem.time >= matched_elem.starttime);
                    uint64_t nic_time = std::max(nextgs[elem.host][elem.nic], elem.time) + g;
                    uint64_t cpu_time = nic_time + lgs_interface->lgs_o + (elem.size - 1) * 0;       
        
                    // satisfy local requires
                    parser.schedules[matched_elem.src].MarkNodeAsDone(matched_elem.offset, cpu_time);
                    check_hosts.insert(matched_elem.src);

                    if (nexto[matched_elem.src][elem.proc] < cpu_time)
                    {
                      nexto[matched_elem.src][elem.proc] = cpu_time;
                    }
                    if (nextgs[matched_elem.src][elem.nic] < nic_time)
                    {
                      nextgs[matched_elem.src][elem.nic] = nic_time;
                    }
                    // check_hosts.push_back(elem.host);
                    parser.schedules[elem.host].MarkNodeAsDone(elem.offset, cpu_time);
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
                    nelem.proc = elem.proc; 
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
                if(0)
                printf("[%i] found msg from %i, t: %lu offset: %i, (CPU: %i)\n", elem.host, elem.target, (ulint)elem.size, elem.offset, elem.proc);
                uint64_t earliestfinish;

                // NUMBER of elements that were searched during message matching
                int32_t match_attempts;
                ruqelem_t matched_elem;
                match_attempts = match(elem, &rq[elem.host], &matched_elem);

                if(match_attempts >= 0) { // found it in RQ
                  
                  if (elem.size == 0) {
                    elem.size = 1;
                  }
                  assert(elem.size > 0);

                  // The message needs to be executed on the CPU that was used to receive the message
                  uint8_t cpu = matched_elem.proc;
                  uint64_t resource_time = std::max(nexto[elem.host][cpu], nextgr[elem.host][elem.nic]);

                  /* printf("OP_MSG found IN RQ -- Resource Time %lu <= Elem Time %lu\n", 
                         resource_time, elem.time); */

                  if (resource_time <= elem.time) {
                    btime_t noise = 0; // No noise in HTSIM
                    //assert(elem.time >= nexto[elem.host][cpu]);

                    uint64_t nic_time = std::max(nextgr[elem.host][elem.nic], elem.time) + g;
                    uint64_t cpu_time = nic_time + 0 + noise + (elem.size-1) * 0;

                    nexto[elem.host][cpu] = cpu_time;
                    nextgr[elem.host][elem.nic] = nic_time;
                    
                    if (print) {
                      printf("-- executing on cpu %i, nexto[%i][%i] = %lu, nextgr[%i][%i] = %lu\n", cpu, elem.host, cpu, nexto[elem.host][cpu], elem.host, elem.nic, nextgr[elem.host][elem.nic]);
                    }

                        // UPDATE the maximum RQ size
                    if(false){ // No Stats collection
                      rq_max[elem.host] = std::max((int)rq[elem.host].size(), rq_max[elem.host]);
                    }

                    if (comm_dep_file_given) {
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


                    // Rend for HTSIM
                    parser.schedules[elem.target].MarkNodeAsDone(elem.offset, cpu_time);
                    //printf("Received1 Something and Matched schedule %d offset %d - Set as Done\n", elem.target, elem.offset);
                    check_hosts.insert(elem.target);

                    if (nexto[elem.target][elem.proc] < cpu_time) {
                      nexto[elem.target][elem.proc] = cpu_time;
                    }
                    if (nextgs[elem.target][elem.nic] < nic_time) {
                      nextgs[elem.target][elem.nic] = nic_time;
                    }


                    // We also need to release the RECV
                    parser.schedules[elem.host].MarkNodeAsDone(matched_elem.offset, cpu_time);
                    //printf("Received2 Something and Matched schedule %d offset %d - Set as Done\n", elem.target, elem.offset);
                    check_hosts.insert(elem.host);
                  } else {
                    if(print)
                      printf("-- msg o,g not available, max(%lu, %lu) > %lu -- reinserting\n",
                            (long unsigned int) nexto[elem.host][cpu],
                            (long unsigned int) nextgr[elem.host][elem.nic],
                            (long unsigned int) elem.time);
                    
                    elem.time = std::max(nexto[elem.host][cpu], nextgr[elem.host][elem.nic]);
                    lgs_interface->aq.push(std::move(elem));
                    // if (cpu_time > nic_time)
                    if (nexto[elem.host][cpu] > nextgr[elem.host][elem.nic])
                      num_reinserts_g++;
                    else
                      num_reinserts_g++;

  #ifdef LIST_MATCH
                    rq[elem.host].push_back(matched_elem);
  #else
                    rq[elem.host][std::make_pair(matched_elem.tag, matched_elem.src)].push(matched_elem);
  #endif
                  }

                } else { // Not in RQ
                  //printf("OP_MSG NOT found IN RQ\n");

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

      // end first while loop
      // do only ask hosts that actually completed something in this round!
      new_events = false;
      // std::sort(check_hosts.begin(), check_hosts.end());
      // check_hosts.erase(unique(check_hosts.begin(), check_hosts.end()), check_hosts.end());
      graph_node_properties recev_msg;
      recev_msg.updated = false;
      bool unlocked_elem = false;

      if (print)
        std::cout << "[INFO] Checking for free operations on hosts:" << std::endl;

      
      for (int host : check_hosts) {

        SerializedGraph *sched=&parser.schedules[host];
  
        // retrieve all free operations
        SerializedGraph::nodelist_t free_ops;
        //printf("Host1 %d - Free Ops Size %d\n", host, free_ops.size());
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
              freeop->time = std::max(lgs_interface->htsim_api->getGlobalTimeNs(), freeop->time);
              if(0)
                printf("Freeop %i (%i,%i) loclop: %lu, time: %lu, offset: %i\n", host, freeop->proc, freeop->nic, (long unsigned int) freeop->size, (long unsigned int)freeop->time, freeop->offset);
              break;
            case OP_SEND:
              freeop->time = std::max(nexto[host][freeop->proc], nextgs[host][freeop->nic]);
              freeop->time = std::max(lgs_interface->htsim_api->getGlobalTimeNs(), freeop->time);
              //freeop->time = lgs_interface->htsim_api->getGlobalTimeNs();
              // if (freeop->proc == 73)
              // {
              //   std::cout << "[DEBUG] HOST " << host << " Send offset " << freeop->offset << " NextO: " << nexto[host][freeop->proc] / 1e9 << " Time: " << freeop->time / 1e9 << std::endl;
              // }
              if(0)
                printf("Freeop %i (%i,%i) send to: %i, tag: %i, size: %lu, time: %lu, offset: %i\n", host, freeop->proc, freeop->nic, freeop->target, freeop->tag, (long unsigned int) freeop->size, (long unsigned int)freeop->time, freeop->offset);
              break;
            case OP_RECV:
              // freeop->time = nexto[host][freeop->proc];
              freeop->time = freeop->starttime;
              if(0)
                printf("Freeop %i (%i,%i) recvs from: %i, tag: %i, size: %lu, time: %lu, offset: %i\n", host, freeop->proc, freeop->nic, freeop->target, freeop->tag, (long unsigned int) freeop->size, (long unsigned int)freeop->time, freeop->offset);
              break;
            default:
              printf("not implemented!\n");
          }
          freeop->time = lgs_interface->htsim_api->getGlobalTimeNs();
          unlocked_elem = true;
          lgs_interface->aq.push(*freeop);
        }
      }

      if (custom_print) {
        printf("1) Sends Active %d - Compute Started %d\n", lgs_interface->sends_active, lgs_interface->compute_started);
        printf("1) Can Simulate Until %" PRId64 " - Top Time %lu\n", can_simulate_until, lgs_interface->aq.top().time);
      }

      bool just_running = false;
      if (lgs_interface->sends_active != 0 || lgs_interface->compute_started != 0) {
          if (!unlocked_elem) {
              lgs_interface->htsim_simulate_until(can_simulate_until);
              just_running = true;
          }
      } 
      if (custom_print) {
        printf("2) Time %lu > htsim time %lu, unlocked_elem %d, just_running %d\n", lgs_interface->aq.top().time, lgs_interface->htsim_api->getGlobalTimeNs(), unlocked_elem, just_running);
      }
      if (lgs_interface->aq.top().time > (lgs_interface->htsim_api->getGlobalTimeNs()) && !unlocked_elem && !just_running) { // Let htim catchup
        lgs_interface->htsim_simulate_until(lgs_interface->aq.top().time);
      }

      if (print)
        std::cout << "[INFO] Finished checking for free operations on hosts. [AQ size: " << lgs_interface->aq.size() << "]" << std::endl;
      if(progress_given) {
        if(num_events/100*lastperc < aqtime) {
          printf("progress: %u %% (%llu) ", lastperc, (unsigned long long)aqtime);
          lastperc++;
          if (lastperc == lgs_interface->percentage_lgs && lgs_interface->percentage_lgs > 0) {
            lgs_interface->terminate_sim();
            exit(0);
          }
          uint maxrq=0, maxuq=0;
          for(uint j=0; j<rq.size(); j++) maxrq=std::max(maxrq, (uint)rq[j].size());
          for(uint j=0; j<uq.size(); j++) maxuq=std::max(maxuq, (uint)uq[j].size());
          printf("[sizes: aq: %i max rq: %u max uq: %u: reinserts: (%lu, %lu, %lu)]\n", (int)lgs_interface->aq.size(), maxrq, maxuq, num_reinserts_o, num_reinserts_g, num_reinserts_net);
          num_reinserts_o=0;
          num_reinserts_g=0;
          num_reinserts_net=0;
        }
      }
    }

    // end second while loop
    gettimeofday(&tend, NULL);
      unsigned long int diff = tend.tv_sec - tstart.tv_sec;

    printf("It terminates! Htsim time %lu\n", lgs_interface->htsim_api->getGlobalTimeNs());
    fflush(stdout);
    
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
      if(p <= 128 && !batchmode_given) { // print all hosts
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
      }  // print only maximum
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
  printf("LGS terminates!\n");
  lgs_interface->terminate_sim();
  fflush(stdout);
  return 0;
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
