/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#ifndef SCHEDGEN_HPP
#define SCHEDGEN_HPP

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
//#include <boost/regex.hpp>
#include "buffer_element.hpp"
#include "cmdline.h"

class Goal;

class LocOp {
private:
  double time_mult; // multiplier - relative to microseconds
public:
  int cpu; // cpu to execute on
  enum t_type {
    IREQU,
    REQU
  }; // type of preceding op (I{send,recv} or {send,recv})
  std::vector<std::pair<int /*Goal::t_id*/, t_type>> prev,
      next;     // preceding and next operations - pairs of id and type
  Goal *goal;   // goal object
  double start; // start time for this local operation

  LocOp(Goal *_goal, double _time_mult, int cpu)
      : time_mult(_time_mult), cpu(cpu), goal(_goal), start(0) {}
  void NextOp(double time, double tend);
};

#include "goal_comm.h"

class Goal {

public:
  Comm *comm;
  typedef int t_id;             // identifier type
  static const t_id NO_ID = -1; // invalid identifier

  typedef std::vector<std::pair<t_id, LocOp::t_type>>
      locop; /* used to identify local operations for dependencies, it's a
                vector of pairs of < id , irequ | requ > */

  Goal(gengetopt_args_info *args_info, int nranks);
  ~Goal();

  void StartOp() { // this starts an operatio
    start.clear();
    end.clear();
  }

  int BuildComm_split(int base_comm, int rank_in_world_comm, int color,
                      int key) {
    Comm *c = this->comm->find_comm(base_comm);
    Comm *nc = c->find_or_create_child_comm(color);
    nc->add_rank_key(rank_in_world_comm, key);
    return nc->getId();
  }

  std::pair<locop, locop> EndOp() {
    locop rstart, rend;
    std::set<t_id>::iterator it;
    for (it = start.begin(); it != start.end(); it++) {
      rstart.push_back(std::make_pair(*it, LocOp::REQU));
    }
    for (it = end.begin(); it != end.end(); it++) {
      rend.push_back(std::make_pair(*it, LocOp::REQU));
    }
    return std::make_pair(rstart, rend);
  }

  void SetTag(uint64_t tag) { curtag = tag; }
  void StartRank(int rank);
  void Comment(std::string c);
  int Send(std::vector<buffer_element> buf, int dest);
  int Send(int size, int dest);
  int Recv(std::vector<buffer_element>, int src);
  int Recv(int size, int src);
  int Exec(std::string opname, btime_t size, int proc);
  int Exec(std::string opname, std::vector<buffer_element> buf);
  int Exec(std::string opname, btime_t size);
  void Requires(int tail, int head);
  void Irequires(int tail, int head);
  void EndRank();
  void Write();
  void AppendString(std::string);

private:
  std::set<t_id> start,
      end; /* the operations which are independent at start and end */
  std::string schedule;
  std::string filename;
  std::fstream myfile;

  /* nonblocking stuff */
  bool nb;
  int poll_int;
  int nbfunc;
  int cpu;
  std::vector<bool> ranks_init;

  t_id id_counter;
  int dummynode;
  int sends, recvs, execs, ranks, reqs;
  uint64_t curtag;

  void read_schedule_from_file();
};

template <typename T> std::vector<T> make_vector(T x) {
  std::vector<T> y;
  y.push_back(x);
  return y;
};

// prototype
void process_trace(gengetopt_args_info *args_info);
void create_binomial_tree_bcast_rank(Goal *goal, int root, int comm_rank,
                                     int comm_size, int datasize);
void create_binomial_tree_reduce_rank(Goal *goal, int root, int comm_rank,
                                      int comm_size, int datasize);
void create_dissemination_rank(Goal *goal, int comm_rank, int comm_size,
                               int datasize);
void create_linear_alltoall_rank(Goal *goal, int src_rank, int comm_size,
                                 int datasize);
void create_allreduce_recdoub_rank(Goal *goal, int src_rank, int comm_size,
                                   int datasize);
void create_allreduce_ring_rank(Goal *goal, int src_rank, int comm_size,
                                int datasize);
#endif
