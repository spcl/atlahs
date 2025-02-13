#include <map>
#include <math.h>
#include <vector>
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <inttypes.h>
#include <graphviz/cgraph.h>

#define noorder 1
#define debug 0

const uint8_t NODE_TYPE_HOST = 1;
const uint8_t NODE_TYPE_SWITCH = 2;


class TGNode {

	friend class TopoGraph;  // no need to write get/set functions for everything (c++ rocks, try this in java...)

	private:
		
		std::vector<TGNode*> ports;            // ports[outport] = ptr to node connected to this port
		std::vector<uint32_t> linkids;        // linkids[port] = the linkid of the link that is connected to port
		std::map</*target*/ uint32_t, uint32_t /*port*/> routinginfo;
		uint8_t type;
		uint32_t id;
	
};

class TopoGraph {
	
	private:
		
		std::vector<TGNode*> nodes;   // nodes[id] = ptr to node with nodeid = id
		std::map<std::string, uint32_t> nodename2nodeid;
		std::map</*linkid*/ uint32_t, /*(nodeid, port)*/ std::pair<uint32_t, uint32_t> > linkid2node_port;
		std::map<uint32_t, uint32_t> rank2nodeid;  //switches also have a nodeid, so we need this map

	public:

		TopoGraph(char* filename) {
			this->parse_dotfile_agraph(filename);
		}

		inline uint32_t create_node(std::string nodename, uint8_t nodetype) {

			/*
				Creates a new node. Nodename must be a unique name, returns the
				id of the created node.
			*/
		
			TGNode* n = new TGNode;
			
			n->type = nodetype;
			n->id = this->nodes.size();
			
			this->nodes.push_back(n);
			this->nodename2nodeid.insert(std::make_pair(nodename, n->id));
		
			return n->id;

		}

		inline uint32_t create_link(std::string from, std::string to) {

			/*
				Creates a new directed link between the nodes "from" and "to". 
				Returns the link id.
			*/
		 
			uint32_t f, t; 
			
			f = (this->nodename2nodeid.find(from))->second;
			t = (this->nodename2nodeid.find(to))->second;
			
			(this->nodes[f])->linkids.push_back(this->linkid2node_port.size());
			this->linkid2node_port.insert(std::make_pair(this->linkid2node_port.size(), std::make_pair((this->nodes[f])->id, (this->nodes[f])->ports.size()) ));
			(this->nodes[f])->ports.push_back(this->nodes[t]);
			
			return (this->linkid2node_port.size()-1);
		}

		inline void add_routing_info(uint32_t dest, uint32_t link) {

			/*
				Adds the target with node-id "dest" to the link with the id
				"link". This means if there is a packet for dest at the node
				at tail(link), this packet will use link to reach the next hop.
			*/
			
			std::map</*linkid*/ uint32_t, /*(nodeid, port)*/ std::pair<uint32_t, uint32_t> >::iterator linkid2node_port_iter;
			linkid2node_port_iter = linkid2node_port.find(link);
			assert(linkid2node_port_iter != linkid2node_port.end());
			uint32_t node = linkid2node_port_iter->second.first;
			uint32_t port = linkid2node_port_iter->second.second;
			nodes[node]->routinginfo.insert(std::make_pair(dest, port));	
		
		}
		
		void parse_dotfile_agraph(char* filename) {
			
			FILE* fd = fopen(filename, "r");
			if (fd == NULL) {
				fprintf(stderr, "Couldn't open %s for reading!\n", filename);
				exit(EXIT_FAILURE);
			}

			Agraph_t* graph = agread(fd, NULL);
			if (graph == NULL) {
				fprintf(stderr, "Couldn't parse graph data!\n");
				fclose(fd);
				exit(EXIT_FAILURE);
			}
			fclose(fd);
			
			// iterate over graphs node's and add them
			Agnode_t* node = agfstnode(graph);
			
			while (node != NULL) {
			
				std::string nodename;
				uint8_t nodetype;
			
				nodename = (std::string) agnameof(node);
				if (nodename.find('H', 0) == 0) nodetype = NODE_TYPE_HOST;
				else nodetype = NODE_TYPE_SWITCH;
				uint32_t nodeid = create_node(nodename, nodetype);
				if (nodetype == NODE_TYPE_HOST) {
					rank2nodeid.insert(std::make_pair(rank2nodeid.size(), nodeid));
				}
				node = agnxtnode(graph, node);

			}

			// now we added all the nodes, so we can add the links between them now
			Agnode_t* node_from = agfstnode(graph);
	
			while (node_from != NULL) {
			
				std::string node_from_name;
			
				node_from_name = (std::string) agnameof(node_from);
				Agedge_t* link = agfstout(graph, node_from);
				
				while (link != NULL) {
					Agnode_t* node_to = aghead(link);
					std::string node_to_name = (std::string) agnameof(node_to);
					uint32_t newlinkid = create_link(node_from_name, node_to_name);
					// add the routinginfo for this link
					char *comment = agget(link, ((char *) "comment"));
					// if the comment is * we leave the routinginfo empty - that means there is only one connection (port 0) and everything goes there
					if (strcmp(comment, "*") != 0) {
    					char* buffer = (char *) malloc(strlen(comment) * sizeof(char) + 1);
						strcpy(buffer, comment);
						char* result = strtok(buffer, ", \t\n");
						while (result != NULL) {
							uint32_t dest = (this->nodename2nodeid.find(std::string(result)))->second;
							add_routing_info(dest, newlinkid);
        					result = strtok(NULL, ", \t\n");
 						}
						free(buffer);
					}
					link = agnxtout(graph, link);
				}
				node_from = agnxtnode(graph, node_from);

			}
			agclose(graph);
		}

	std::vector</*linkids*/ uint32_t> find_route(uint32_t src_rank, uint32_t dest_rank) {
		
		std::map<uint32_t, uint32_t>::iterator r2nid_it;
		r2nid_it = rank2nodeid.find(src_rank);
		assert(r2nid_it != rank2nodeid.end());
		uint32_t src_id = r2nid_it->second;
		r2nid_it = rank2nodeid.find(dest_rank);
		assert(r2nid_it != rank2nodeid.end());
		uint32_t dest_id = r2nid_it->second;

		assert(nodes.size() > dest_id);
		std::vector<uint32_t> path;
		uint32_t pos = src_id;
		while (pos != dest_id) {
			assert(nodes.size() > pos);
			assert(nodes[pos]->ports.size() > 0);
			if (nodes[pos]->ports.size() == 1) {
				// there is only one outgoing link, use it
				path.push_back(nodes[pos]->linkids[0]);
				pos = nodes[pos]->ports[0]->id;
			}
			else {
				// there are multiple outgoing ports - find the right one
				std::map</*target*/ uint32_t, uint32_t /*port*/>::iterator nexthop;
				nexthop = nodes[pos]->routinginfo.find(dest_id);
				assert(nexthop != nodes[pos]->routinginfo.end());
				path.push_back(nodes[pos]->linkids[nexthop->second]);
				pos = nodes[pos]->ports[nexthop->second]->id;
			}
		}
		return path;
	}

};

class SimpleNetwork {
	
	private:

	TopoGraph* topograph;

	struct msg_data {
		int64_t bytes_left; // how many bytes have yet to be transmitted
		uint64_t lasttime;   // the last time we adjusted bytes_left
		uint32_t src;        // source of that message
		uint32_t dst;        // destination of that message
		uint32_t max_cong;   // congestion for that message
	};

	typedef std::map< /*msg_id*/ uint32_t, /*message data*/ msg_data*> msg_db_t;
	msg_db_t msg_db;

	typedef std::map< /*link_id*/ uint32_t, /*msgs using this link*/ std::vector<uint32_t> > lnk_db_t;
	lnk_db_t lnk_db;

#if noorder
	
	/*
		This is only needed if we can not make sure that for every timestep,
		all queries will happen before any insert
	*/
	
	/*
		TODO rethink which datastructures are used
	*/

	typedef std::map< /*time*/ uint64_t, std::map</*msg_id*/ uint32_t, /*dummy*/ uint8_t>  > querry_db_t;
	querry_db_t querry_db;
	typedef std::map</*msg_handle*/ uint32_t, /*dummy*/ uint8_t>  finished_msgs_db_t;
	finished_msgs_db_t finished_msgs;

#endif

	inline std::vector<uint32_t> get_route(uint32_t src, uint32_t dest) {
		/**
			This function returns a vector of edge ids, which contains all edges 
			travelled by a message going from src to dest
		*/

		std::vector<uint32_t> path;
		
		path = topograph->find_route(src, dest);	
		
		return path;
	}

	public:

  SimpleNetwork(char* filename) {
    // htor: read network topology here
	topograph = new TopoGraph(filename);
  }

	uint64_t query(uint64_t starttime, uint64_t currtime, uint32_t src, uint32_t dest, uint64_t size, uint32_t* msg_handle) {
	
		/**
			Checks if the message that was started at starttime was delayed since then.
			Returns new endtime. If endtime==curtime, all congestion caused by this message is freed.
		*/
		// get the queried message
		msg_db_t::iterator msg_db_it = msg_db.find(*msg_handle);
		assert(msg_db_it != msg_db.end());

#if noorder
		// check if the message is already in the finished msgs db
		finished_msgs_db_t::iterator fmdbit;
		fmdbit = finished_msgs.find(*msg_handle);
		if (fmdbit != finished_msgs.end()) {
			// The message is allready finished - delete it from the finished msgs db and return
			finished_msgs.erase(fmdbit);
			return currtime;
#if debug
			printf("\tThis msg was already finished.\n");
#endif
		}
#endif
		// adjust bytes_left
		uint64_t dt = currtime - msg_db_it->second->lasttime;
		msg_db_it->second->bytes_left -= (uint64_t) floor((double) (dt*1e4) / msg_db_it->second->max_cong);
		msg_db_it->second->lasttime = currtime;
		// check if the message is finished
		if (msg_db_it->second->bytes_left <= 0) {
			// remove all information associated with that message
      		delete msg_db_it->second; // htor - I hope it's save to delete it here (was a mem leak)
			msg_db.erase(msg_db_it);
			std::vector<uint32_t> path = get_route(src, dest);
			std::map<uint32_t, char> recalc;
			std::vector<uint32_t>::iterator e_i;
			for (e_i = path.begin(); e_i != path.end(); e_i++) {
				lnk_db_t::iterator lnk_db_it = lnk_db.find(*e_i);
				assert(lnk_db_it != lnk_db.end());
				// adjust bytes_left and lasttime for each message using this link or mark the msg for deletion
				// (immediate deletion would invalidate the iterator)
				std::vector<uint32_t>::iterator msg_id, msg_delhandle;
				for (msg_id = lnk_db_it->second.begin(); msg_id != lnk_db_it->second.end(); msg_id++) { 
					if (*msg_id == *msg_handle) {
						msg_delhandle = msg_id;
					}
					else {
						// save this me id we need to recalculate its max_cong value - we save into a map, so we dont recalc the same msg twice
						recalc.insert(std::make_pair(*msg_id, '1')); // the second param is useless
						msg_db_it = msg_db.find(*msg_id);
						assert(msg_db_it != msg_db.end());
						uint64_t dt = currtime - msg_db_it->second->lasttime;
						//if (msg_db_it->second->max_cong > 1) {printf("Congestion of %i\n", msg_db_it->second->max_cong);}
						msg_db_it->second->bytes_left -= (uint64_t) floor((double) (dt*1e4) / msg_db_it->second->max_cong);
						msg_db_it->second->lasttime = currtime;
					}

				}
				// delete the old message from the list of messages using this link
				lnk_db_it->second.erase(msg_delhandle);
			}
			// now recalculate all the max_cong values that might need recalculation
			std::map<uint32_t, char>::iterator recalc_it;
			for (recalc_it = recalc.begin(); recalc_it != recalc.end(); recalc_it++) {
				msg_db_it = msg_db.find(recalc_it->first);
				// get the path for this message
				std::vector<uint32_t> path = get_route(msg_db_it->second->src, msg_db_it->second->dst);
				// go along the path, get new max_cong (the finished message is already deleted)
				msg_db_it->second->max_cong = 0;
				for (e_i = path.begin(); e_i != path.end(); e_i++) {
					lnk_db_t::iterator lnk_db_it = lnk_db.find(*e_i);
					if (lnk_db_it->second.size() > msg_db_it->second->max_cong) {
						msg_db_it->second->max_cong = lnk_db_it->second.size();
					}
				}
			}
#if debug
			printf("\t msg finished\n");
#endif
			return currtime;
		}
		else {
			// the message is not finished yet, return the next time the message could possibly be finished
#if debug
			printf("\t msg still in transfer, bytes left: %lu\n", (long unsigned int) msg_db_it->second->bytes_left );
#endif
			return currtime + ceil( (double) msg_db_it->second->bytes_left / 1e4);
		}

	}

	uint64_t insert(uint64_t currtime, uint32_t src, uint32_t dest, uint64_t size, uint32_t *msg_handle) {

		
		/**
			Computes the time needed to forward a message of size bytes from
			src to dest with the current congestion. Returns estimated endtime.
			Complexity: \Theta(pathlen * avg(number of msgs on edges in path))

			msg_handle* will be set to a value needed by the query function to
			retrieve the message.
		*/

#if noorder	
		/**
			It might be difficult to ensure that all queries happen before the
			inserts. So we maintain our own database of messages that need
			querying for now
		*/

		// check if the query db contains messages for currtime, if yes, querry them
		// if the are finished put them in the finished_db

		querry_db_t::iterator qit = querry_db.find(currtime);
		if (qit != querry_db.end()) {
			std::map</*msg id*/ uint32_t, /*dummy*/ uint8_t>::iterator qmit;
			for (qmit = qit->second.begin(); qmit != qit->second.end(); qmit++) {
						
				// get the message id to querry  message
				msg_db_t::iterator msg_db_it = msg_db.find(qmit->first);
				assert(msg_db_it != msg_db.end());
				// querry the message
				uint32_t mh = qmit->first;
				uint64_t querryres = query(currtime, currtime, msg_db_it->second->src, msg_db_it->second->dst, 0, &mh);
				if (querryres == currtime) {
					// the message is finished! add it to the finished msgs db
					finished_msgs.insert(std::make_pair(qmit->first, 1));
				}
			}
		}
#endif


		static uint32_t msg_index = 0;

		std::vector<uint32_t> path = get_route(src, dest);
	
		// add the new message to the msg_db
		msg_data *nmsg = new msg_data;
		assert(size > 0);
		nmsg->bytes_left = size*1e4;	// this is not an integer since we divide by max. cong, so we emulate fp with the multiplication with 1e4
		nmsg->lasttime = currtime;
		nmsg->src = src;
		nmsg->dst = dest;
		nmsg->max_cong = 1; // only stays this way if the path of this message is empty
		*msg_handle = msg_index; 
		msg_db.insert(std::make_pair(msg_index, nmsg));

		
		// update bytes_left and lasttime for every message along the new messages path, since we might change their congestion.
		std::vector<uint32_t>::iterator e_i;
		for (e_i = path.begin(); e_i != path.end(); e_i++) {
			// get messages on e_i if available
			lnk_db_t::iterator lnk_db_it = lnk_db.find(*e_i);
			// check if e_i is already in the edge db
			if (lnk_db_it != lnk_db.end()) {
				std::vector<uint32_t>::iterator msg_id;
				
				 // keep track of max cong along path
				if (nmsg->max_cong < lnk_db_it->second.size()+1) nmsg->max_cong = lnk_db_it->second.size() + 1;
				
				for (msg_id = lnk_db_it->second.begin(); msg_id != lnk_db_it->second.end(); msg_id++) {
					msg_db_t::iterator msg_db_it = msg_db.find(*msg_id);
					assert(msg_db_it != msg_db.end());
					uint64_t dt = currtime - msg_db_it->second->lasttime;
					msg_db_it->second->bytes_left -= (uint64_t) floor((double) (dt*1e4) / msg_db_it->second->max_cong); // TODO: incorporate G somehow (now: 1b / timeslice)
					assert(msg_db_it->second->bytes_left > 0);
					msg_db_it->second->lasttime = currtime;
					if (msg_db_it->second->max_cong == lnk_db_it->second.size()) {
						// if this message (msg_db_it) is experiencing max_cong on e_i, increase it's max_cong because the new message will also use this edge
						msg_db_it->second->max_cong++;
					}
				}
				// add the new message to the messages that go through e_i 
				lnk_db_it->second.push_back(msg_index);
			}
			// if this edge wasn't in the db yet, add it, also add the new massage to the messages that use it
			else {
				std::vector<uint32_t> msgs;
				msgs.push_back(msg_index);
				lnk_db.insert( make_pair(*e_i, msgs) );
			}
		}
		msg_index++;
		uint64_t retval = currtime + size; // return the earliest possible time at which this message can finish (time without congestion) TODO: incorporate G
#if noorder
		// add this message into the "querry db"
	    // std::map< /*time*/ uint64_t, std::map</*msg_id*/ uint32_t, /*dummy*/ uint8_t>  > querry_db;
		
		// first we need to check if there is already an entry for retval - the time this (new) message will finish
		querry_db_t::iterator qdbit = querry_db.find(retval);
		if (qdbit == querry_db.end()) {
			// there was no such entry, create one
			std::map</*msg_id*/ uint32_t, /*dummy*/ uint8_t> msglist;
			msglist.insert(std::make_pair(*msg_handle, 0));
			querry_db.insert(std::make_pair(retval, msglist));
		}
		else {
			// there already was an entry, just add our new msg to the map
			qdbit->second.insert(std::make_pair(*msg_handle, 0));
		}

		// now we should also clean the querry db, all querry reqs with time < currtime can be deleted
		// I do this on inserts instead of querries because inserts are less frequent

		std::vector<uint64_t> mark_for_del;
		for (querry_db_t::iterator qdbit = querry_db.begin(); qdbit != querry_db.end(); qdbit++) {
			if (qdbit->first < currtime) mark_for_del.push_back(qdbit->first);
		}
		for (std::vector<uint64_t>::iterator mfdit = mark_for_del.begin(); mfdit != mark_for_del.end(); mfdit++) {
			querry_db_t::iterator qdbit;
			qdbit = querry_db.find(*mfdit);
			assert(qdbit != querry_db.end());
			querry_db.erase(qdbit);
		}

#endif
	
		return retval;
	}

};



// this is a simple dispatcher class!
class Network {
  private:
    SimpleNetwork *simplenet;
    int type;
    static const int LOGGP=0;
    static const int SIMPLE=1;
  public:
  
  Network(gengetopt_args_info *args_info) {
    type = LOGGP;
    if(strcmp(args_info->network_type_arg, "LogGP") == 0) {
      // this is a total dummy class (doesn't even exist ;))
      type = LOGGP;
      std::cout << "LogGP network backend; ";
    } else if(strcmp(args_info->network_type_arg, "simple") == 0) {
      simplenet = new SimpleNetwork(args_info->network_file_arg);
      type = SIMPLE;
      std::cout << "Simple network backend; ";
    }
  }

	uint64_t insert(uint64_t currtime, uint32_t src, uint32_t dest, uint64_t size, uint32_t *msg_handle) {
#if debug
		printf("insert called by simulator - insert(currtime %lu, src %u, dest %u, size %lu)\n", (long unsigned int) currtime, 
																								 (unsigned int) src, 
																								 (unsigned int) dest,
																								 (long unsigned int) size);
#endif
		if(type == SIMPLE) {
			uint64_t retval = simplenet->insert(currtime, src, dest, size, msg_handle);
#if debug
			printf("insert returned %lu, msg-id is %u\n", (long unsigned int) retval, *((unsigned int*)msg_handle));
#else
      retval=0; // shot off stupid compiler warning, will be optimized out anyway (hopefully)
#endif
		}
		return currtime;
	}

	uint64_t query(uint64_t starttime, uint64_t currtime, uint32_t src, uint32_t dest, uint64_t size, uint32_t* msg_handle) {
#if debug
		printf("query called by simulator - query(starttime %lu, currtime %lu, src %u, dest %u, size %lu, msg-id %u)\n", (long unsigned int) starttime,
																														 (long unsigned int) currtime,
																														 (unsigned int) src, 
																														 (unsigned int) dest,
																														 (long unsigned int) size, 
																														 *((unsigned int*)msg_handle) );
#endif
	if(type == SIMPLE) {
			uint64_t retval = simplenet->query(starttime, currtime, src, dest, size, msg_handle);
//#if debug
			printf("query returned %lu\n", (long unsigned int) retval);
//#endif
			return retval;
		}
    	return currtime;
	}
  
};
