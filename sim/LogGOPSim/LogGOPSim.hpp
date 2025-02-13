#include <inttypes.h>

#ifndef GRAPH_NODE_PROPERTIES
#define GRAPH_NODE_PROPERTIES 1

typedef uint64_t btime_t;
  
/* this class is CRITICAL -- keep it as SMALL as possible! 
 *
 * current size: 39 bytes
 *
 */
class graph_node_properties {
	public:
    btime_t time;
    btime_t starttime;         // only used for MSGs to identify start times
#ifdef HOSTSYNC
    btime_t syncstart;
#endif
#ifdef STRICT_ORDER
    btime_t ts; /* this is a timestamp that determines the (original) insertion order of 
                  elements in the queue, it is increased for every new element, not for 
                  re-insertions! Needed for correctness. */
#endif
		uint64_t size;						// number of bytes to send, recv, or time to spend in loclop
		uint32_t target;					// partner for send/recv
    uint32_t host;            // owning host 
		uint32_t offset;          // for Parser (to identify schedule element)
		uint32_t tag;							// tag for send/recv
    uint32_t handle;          // handle for network layer :-/
		uint8_t proc;							// processing element for this operation
		uint8_t nic;							// network interface for this operation
		char type;							  // see below
};

/* this is a comparison functor that can be used to compare and sort
 * operation types of graph_node_properties */
class gnp_op_comp_func {
  public:
  bool operator()(graph_node_properties x, graph_node_properties y) {
    if(x.type < y.type) return true;
    return false;
  }
};

/* this is a comparison functor that can be used to compare and sort
 * graph_node_properties by time */
class aqcompare_func {
  public:
  bool operator()(graph_node_properties x, graph_node_properties y) {
    if(x.time > y.time) return true;
#ifdef STRICT_ORDER
    if(x.time == y.time && x.ts > y.ts) return true; 
#endif
    return false;
  }
};


// mnemonic defines for op type
static const int OP_SEND = 1;
static const int OP_RECV = 2;
static const int OP_LOCOP = 3;
static const int OP_MSG = 4;
		
static const uint32_t ANY_SOURCE = ~0;
static const uint32_t ANY_TAG = ~0;


#endif
