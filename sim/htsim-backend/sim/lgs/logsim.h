/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef LOGSIM_H
#define LOGSIM_H
#include <inttypes.h>
#include <list>

#ifndef GRAPH_NODE_PROPERTIES
#define GRAPH_NODE_PROPERTIES 1
#define STRICT_ORDER // this is needed to keep order between Send/Recv and
                     // LocalOps in NBC case :-/

#define STRICT_ORDER // this is needed to keep order between Send/Recv and
                     // LocalOps in NBC case :-/
#define LIST_MATCH   // enables debugging the queues (check if empty)
#define HOSTSYNC     // this is experimental to count synchronization times induced
                     // by message transmissions

#include <algorithm>
#include <iostream>
#include <list>
#include <stdio.h>
#include <stdlib.h>

#ifndef LIST_MATCH
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif
#endif

typedef uint64_t btime_t;

/* ... */
// void schedule_event(int);
class graph_node_properties {
  public:
    btime_t time;
    btime_t starttime; // only used for MSGs to identify start times
#ifdef HOSTSYNC
    btime_t syncstart;
#endif
#ifdef STRICT_ORDER
    btime_t ts; /* this is a timestamp that determines the (original) insertion
                  order of elements in the queue, it is increased for every new
                  element, not for re-insertions! Needed for correctness. */
#endif
    uint64_t size;   // number of bytes to send, recv, or time to spend in loclop
    uint32_t target; // partner for send/recv
    uint32_t host;   // owning host
    uint32_t offset; // for Parser (to identify schedule element)
    uint32_t tag;    // tag for send/recv
    uint32_t handle; // handle for network layer :-/
    uint8_t proc;    // processing element for this operation
    uint8_t nic;     // network interface for this operation
    char type;       // see below
    bool updated = false;
};

/* this is a comparison functor that can be used to compare and sort
 * operation types of graph_node_properties */
class gnp_op_comp_func {
  public:
    bool operator()(graph_node_properties x, graph_node_properties y) {
        if (x.type < y.type)
            return true;
        return false;
    }
};

/* this is a comparison functor that can be used to compare and sort
 * graph_node_properties by time */
class aqcompare_func {
  public:
    bool operator()(graph_node_properties x, graph_node_properties y) {
        if (x.time > y.time)
            return true;
#ifdef STRICT_ORDER
        if (x.time == y.time && x.ts > y.ts)
            return true;
#endif
        return false;
    }
};

// mnemonic defines for op type
static const int OP_SEND = 1;
static const int OP_RECV = 2;
static const int OP_LOCOP = 3;
static const int OP_LOCOP_IN_PROGRESS = 4;
static const int OP_MSG = 5;
static const int OP_TIME = 6;

static const uint32_t ANY_SOURCE = ~0;
static const uint32_t ANY_TAG = ~0;
// bool last_elem_op_msg = false;

//}
#endif

typedef struct {
    // TODO: src and tag can go in case of hashmap matching
    btime_t starttime; // only for visualization
    uint32_t size, src, tag, offset;
} ruqelem_t;
typedef std::list<ruqelem_t> ruq_t;

typedef unsigned int uint;
typedef unsigned long int ulint;
typedef unsigned long long int ullint;

#ifdef LIST_MATCH
// TODO this is really slow - reconsider design of rq and uq!
// matches and removes element from list if found, otherwise returns
// false
static inline int match(const graph_node_properties &elem, ruq_t *q, ruqelem_t *retelem = NULL) {

    // MATCH attempts (i.e., number of elements searched to find a matching
    // element)
    int match_attempts = 0;

    // std::cout << "UQ size " << q->size() << "\n";

    if (0)
        printf("++ size is %lu,  host [%i] searching matching queue for src "
               "target %i "
               "tag "
               "%i\n",
               q->size(), elem.host, elem.target, elem.tag);
    for (ruq_t::iterator iter = q->begin(); iter != q->end(); ++iter) {
        match_attempts++;
        if (0)
            printf("Compared element is -> %d %d vs %d %d\n", iter->src, iter->tag, elem.target, elem.tag);
        if (elem.target == ANY_SOURCE || iter->src == ANY_SOURCE || iter->src == elem.target) {
            if (elem.tag == ANY_TAG || iter->tag == ANY_TAG || iter->tag == elem.tag) {
                if (retelem)
                    *retelem = *iter;
                q->erase(iter);
                return match_attempts;
            }
        }
    }
    return -1;
}
#else
class myhash { // I WANT LAMBDAS! :)
  public:
    size_t operator()(const std::pair<int, int> &x) const { return (x.first >> 16) + x.second; }
};
typedef std::hash_map<std::pair</*tag*/ int, int /*src*/>, std::queue<ruqelem_t>, myhash> ruq_t;
static inline int match(const graph_node_properties &elem, ruq_t *q, ruqelem_t *retelem = NULL) {

    if (print)
        printf("++ [%i] searching matching queue for src %i tag %i\n", elem.host, elem.target, elem.tag);

    ruq_t::iterator iter = q->find(std::make_pair(elem.tag, elem.target));
    if (iter == q->end()) {
        return -1;
    }
    std::queue<ruqelem_t> *tq = &iter->second;
    if (tq->empty()) {
        return -1;
    }
    if (retelem)
        *retelem = tq->front();
    tq->pop();
    return 0;
}
#endif

int size_queue(std::vector<ruq_t> my_queue, int num_proce);
#endif /* LOGSIM_H */
