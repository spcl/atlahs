/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include "trace_reader.hpp"

#include <assert.h>
#include <map>
#include <math.h>
#include <stdexcept>

// The init operation of a persistent request
// takes a certain amount of time, which is artificially
// increased during tracing, this multiplier is used to
// correct the time. The smaller it is, the more time
// is added to the init operation.
// Needs to be calibrated for different machines
#define REQ_INIT_MULTIPLIER 0.9
std::map<uint32_t, uint8_t> comm_map;
uint8_t comm_index = 0;

/**
 * A struct to represent a persistent request
 **/
typedef struct {
  unsigned long req_id;
  bool is_send;
  // Target rank, can be either source or destination
  uint32_t target;
  // The tag of the request
  int tag;
  int comm;
  uint32_t size;
} p_request_t;

uint32_t MAKE_TAG(int comm, int tag) 
{
  uint32_t comm_id = (uint32_t) comm;
  if (comm_map.count(comm))
  {
    // If 'comm' exists
    comm_id = comm_map.find(comm)->second;
  }
  else
  {
    // if 'comm' doesn't exist
    comm_map[comm] = comm_index;
    comm_id = comm_index++;
  }

  if (comm_id > 256)
  {
    fprintf(stderr, "comm too big\n");
  }

  if (tag > (1 << 23))
  {
    fprintf(stderr, "tag too big\n");
    std::cout << "[ERROR] tag " << tag << std::endl;
  }
  /* std::cout << "[DEBUG] comm: " << comm << ", comm_id: " << comm_id <<
      ", tag: " << tag << std::endl; */
  comm_id = comm_id << 24;
  tag = comm_id | tag;
  return tag;
}

// int MAKE_TAG(int comm, int tag) {
//   if (comm > 256)
//     fprintf(stderr, "comm too big\n");
//   if (tag > (1 << 23))
//     fprintf(stderr, "tag too big\n");
//   comm = comm << 24;
//   tag = comm | tag;
//   return tag;
// }

// TODO: dangerous
static const int req_size = 4;
static const int mpi_any_source = -2;
static const int collsbase = 1000000;
static const int MAX_STRLEN = 4096;

// print debug (overridden by cmdline argument)
static int print = 0;

// arrays of MPI functions to match, may not contain '\0' or ':' in function
// name!
static const char *mpifuncs[] = {
    "MPI_Allreduce", "MPI_Iallreduce", "MPI_Alltoall",  "MPI_Bcast",
    "MPI_Barrier",   "MPI_Init",       "MPI_Finalize",  "MPI_Alltoallv",
    "MPI_Scatter",   "MPI_Scatterv",   "MPI_Gather",    "MPI_Gatherv",
    "MPI_Allgather", "MPI_Allgatherv", "MPI_Reduce",    "MPI_Irecv",
    "MPI_Send",      "MPI_Recv",       "MPI_Comm_rank", "MPI_Comm_size",
    "MPI_Isend",     "MPI_Wait",       "MPI_Waitall",   "MPI_Iprobe",
    "MPI_Testall",   "MPI_Test",       "MPI_Scan",      "MPI_Exscan",
    "MPI_Get_count", "MPI_Sendrecv",   "MPI_Rsend",     "MPI_Init_thread",
    // Persistent requests
    "MPI_Send_init", "MPI_Recv_init",  "MPI_Start",     "MPI_Request_free",
    /* end marker */ "\0"};

class htorMatcher;
class htorMatch;
class htorParser {
private:
  char *line;
  char *elements;

public:
  int hashlength;
  std::map<int, int> hash2pos;

  int hashFunc(char *string, int chars) {
    int hash = 0;
    // hash function name up to minchars
    for (int pos = 0; pos < chars; ++pos) {
      // chars was too long - ran over end of string!
      if (string[pos] == '\0')
        break;
      if (string[pos] == ':')
        break;

      hash += (pos + 1) * (string[pos] >> 1);
    }
    return hash;
  }

  htorParser() {
    /* simple hash function - sum (char at pos i * i)
     * check if it is collision-free */

    /* find maximum lengh string in mpifuncs */
    int maxlen = 0;
    for (int func = 0; mpifuncs[func][0] != '\0';
         func++) { // iterate over all function names
      if ((int)strlen(mpifuncs[func]) > maxlen)
        maxlen = strlen(mpifuncs[func]);
    }

    int minchars;
    for (minchars = 1; minchars <= maxlen;
         minchars++) { // minimal needed number of chars to distinguish all
                       // functions!
      bool found = false;
      hash2pos.clear();
      for (int func = 0; mpifuncs[func][0] != '\0';
           func++) { // iterate over all function names
        int hash = hashFunc((char *)mpifuncs[func], minchars);
        // if(hash == -1) break;
        // std::cout << mpifuncs[func] << " hash (" << minchars << "): " << hash
        // << "\n";

        // see if we already had this hash
        std::map<int, int>::iterator iter = hash2pos.find(hash);
        if (iter != hash2pos.end()) {
          found = true;
          break;
        }

        hash2pos[hash] = func;
      }
      // check if all values in "hashes" are unique
      // I know, this is O(n^2), but the input is constant!
      if (found == false)
        break;
    }
    if (minchars <= maxlen) {
      if (print)
        std::cout << "initialized hash-searcher to " << minchars << "\n";
      hashlength = minchars;
    } else {
      std::cerr << "failed to initialized hash-searcher (" << minchars
                << ") - change hash algorithm or check mpifuncs for doubles!\n";
      throw(130);
    }
  }

  // matches a string and returns the number of elements that has been found
  bool match(const class htorMatcher *matcher, int hash, char *line,
             class htorMatch *match);
  bool match(const class htorMatcher *matcher, char *line, class htorMatch *m) {
    // get hast for line
    int hash = hashFunc(line, hashlength);
    return match(matcher, hash, line, m);
  }
};

/* this is to statically precompute all hashes! */
class htorMatcher {
public:
  int myhash;
  int len;

  htorMatcher(class htorParser *parser, const char *str) {
    len = strlen(str);
    // assert(len >= parser->hashlength);

    myhash = parser->hashFunc((char *)str, parser->hashlength);
    assert(myhash != -1);

    std::map<int, int>::iterator iter = parser->hash2pos.find(myhash);
    if (iter == parser->hash2pos.end()) {
      std::cerr << "func with hash '" << myhash
                << "' not found! Check mpifuncs array!\n";
      throw(130);
    }
  }
};

class htorMatch {
public:
  std::vector<int> offsets;
  char *line;
  char saved;

  // this is evil! This changes the next delimiter symbol to '\0' if
  // init == 1 and changes it back to the saved value if init == 0
  void prepOffsets(int pos, int init) {
    if (init == 1) {
      saved = offsets[pos + 1];
      line[offsets[pos + 1]] = '\0';
    } else {
      line[offsets[pos + 1]] = saved;
    }
  }

  void get(int pos, int *y) {
    pos--; // compatibility to boost::regexp
    prepOffsets(pos, 1);
    sscanf(&line[offsets[pos]] + 1, "%i", y);
    prepOffsets(pos, 0);
    // printf("%i\n", *y);
  }

  void get(int pos, unsigned long *y) {
    pos--; // compatibility to boost::regexp
    prepOffsets(pos, 1);
    sscanf(&line[offsets[pos]] + 1, "%lu", y);
    prepOffsets(pos, 0);
    // printf("%i\n", *y);
  }

  void get(int pos, double *y) {
    pos--; // compatibility to boost::regexp
    prepOffsets(pos, 1);
    sscanf(&line[offsets[pos]] + 1, "%lf", y);
    prepOffsets(pos, 0);
    // printf("%f\n", *y);
  }
};

bool htorParser::match(const class htorMatcher *matcher, int hash, char *line,
                       class htorMatch *match) {

  // std::cout << hash << " " << line << "\n";

  if (hash == matcher->myhash) {
    if (strncmp("MPI_", line, 4) != 0) {
      fprintf(stderr,
              "line [%s] does not start with MPI_, this might lead to errors\n",
              line);
      exit(-1);
    }
    // std::cout << "matched: ";
    // std::cout << line << "\n";

    match->offsets.clear();
    match->line = line;

    // start at later pos
    int pos = 0;
    while (line[pos] != '\0') {
      if (line[pos] == ':') {
        // std::cout << "found : at: " << pos << "\n";
        match->offsets.push_back(pos);
      }
      if (line[pos] == ',') {
        // std::cout << "found , at: " << pos << "\n";
        match->offsets.push_back(pos);
      }
      pos++;
      // std::cout << ">" << line[pos] << "<\n";
    }
    match->offsets.push_back(pos);

    return true;
  }

  return false;
}

void LocOp::NextOp(double time, double end) {

  time -= this->start;
  if (time < 0) {
    std::cout << "negative operation time (time=" << time
              << " this->start=" << this->start << " end=" << end << ")"
              << std::endl;
  }

  if (print)
    printf(" loclop: %llu\n", (unsigned long long)(round(time * time_mult)));
  int op = this->goal->Exec(
      "comp", (unsigned long long)(round(time * time_mult)), this->cpu);

  // operations that depend on me
  std::vector<std::pair<Goal::t_id, t_type>>::iterator it;
  for (it = this->next.begin(); it != this->next.end(); it++) {
    if (it->first != Goal::NO_ID) {
      if (it->second == REQU)
        this->goal->Requires(it->first, op);
      else
        this->goal->Irequires(it->first, op);
    }
  }
  // operations that I depend on
  for (it = this->prev.begin(); it != this->prev.end(); it++) {
    if (it->first != Goal::NO_ID) {
      if (it->second == REQU)
        this->goal->Requires(op, it->first);
      else
        this->goal->Irequires(op, it->first);
    }
  }
  this->prev.clear();
  this->next.clear();

  // start new locop
  // this->start=this->start+time;
  this->start = end;
}

// get \ceil log_base(i) \ceil with integer arithmetic
int logi(int base, int x) {
  int log = 0;
  int y = 1;
  while (y <= x) {
    log++;
    y *= base;
  }
  return log;
}

// pretty print numbers in buffer (add 0's to fill up to max)
int pprint(char *buf, int len, int x, int max) {
  int log10x = logi(10, x);
  if (x == 0)
    log10x = 1; // log_x(0) is undefined but has a single digit ;)
  int log10max = logi(10, max);
  int i;
  for (i = 0; i < log10max - log10x; ++i) {
    *buf = '0';
    buf++;
    len--;
  }
  snprintf(buf, len, "%i", x);
  return log10max - log10x;
}

void change_zero_to_host(char *mask, char *buffer, int host) {
  char *substr = strstr(mask, "-0");
  if (substr == NULL) {
    std::cerr << "tracefile-name did not contain '-0' - exiting\n";
    throw(130);
  }
  substr = strstr(substr + 1, "-0");
  if (substr != NULL) {
    std::cerr << "tracefile-name did contain more than one '-0' - exiting\n";
    throw(130);
  }

  char str_start[MAX_STRLEN];
  char str_end[MAX_STRLEN];

  // extract everything before the 0
  strcpy(str_start, mask);
  substr = strstr(str_start, "-0");
  substr++; // go over '-'
  *substr = '\0';

  // extract everything after the 0
  strcpy(str_end, mask);
  substr = strstr(str_end, "-0");
  substr++; // go over '-'
  while (*substr == '0')
    substr++;
  strcpy(str_end, substr);

  sprintf(buffer, "%s%i%s", str_start, host, str_end);
}

static inline Goal::t_id
finish_coll(std::string collname /* the op id */,
            std::pair<Goal::locop, Goal::locop> ops /* dependent operations */,
            double tstart, double tend, int nbcify, LocOp *curlocop,
            Goal *goal) {

  Goal::locop::iterator it;

  Goal::t_id collop = goal->Exec(collname.c_str(), 0);

  // an operation represents the time *before* the current collective
  // it is initialized with curlocop->time with the finishing time of
  // the last one (or 0 respectively)

  // the collop depends on all last guys from the collective
  // this operation represents the whole collective!
  for (it = ops.second.begin(); it != ops.second.end(); it++) {
    goal->Requires(collop, it->first);
  }

  // the independent ops in the collective must all depend on the
  // current localop (we could also introduce a virtual dependency here)
  for (it = ops.first.begin(); it != ops.first.end(); it++) {
    curlocop->next.push_back(*it);
  }

  // shorten the last localop
  double virtstart = tstart - nbcify;
  if (virtstart - curlocop->start < 0) {
    virtstart = curlocop->start;
    if (print)
      std::cout << "nbcify shortened from " << nbcify << " to "
                << tstart - virtstart << "\n";
    nbcify = tstart - virtstart;
  }

  // create new NBC localop
  Goal::t_id nbcop = Goal::NO_ID;
  if (nbcify) {
    nbcop = goal->Exec("nbcify", nbcify, curlocop->cpu);
    goal->Irequires(nbcop, collop);
  }

  // finish last localop
  curlocop->NextOp(virtstart, tend);

  // this is for the next localop! (which can only happen after the
  // current collective ends
  curlocop->prev = make_vector(std::make_pair(collop, LocOp::REQU));
  return collop;
}

void process_trace(gengetopt_args_info *args_info) {

  if (!args_info->traces_given) {
    std::cout << "please give me tracefiles ;)" << std::endl;
    return;
  }

  // iterate over all possible hosts
  int host = -1;

  print = args_info->traces_print_arg;

  std::cout << "using file mask: " << args_info->traces_arg << std::endl;
  std::string fptrn(args_info->traces_arg);

  /* get the number of files (hosts) - same as below - this just counts the
   * commsize */
  while (1) {
    char buffer[MAX_STRLEN];
    change_zero_to_host(args_info->traces_arg, buffer, ++host);
    assert(strlen(buffer) < MAX_STRLEN);

    std::ifstream trace(buffer, std::ios::in);
    if (!trace.is_open())
      break;
  }
  int hosts = host;

  int nbcify = args_info->traces_nbcify_arg;
  int extrhosts = args_info->traces_extr_arg;

  std::cout << "found " << hosts << " hosts, extrapolating to  "
            << extrhosts * hosts << std::endl;
  std::cout << "timebase: " << args_info->timemult_arg << "; using CPU "
            << args_info->cpu_arg << " for computation\n";
  if (nbcify)
    std::cout << "nbcify propost: " << nbcify << "\n";

  Goal goal(args_info, hosts * extrhosts);

  /* see if we have a file with start lines for the trace files - one
   * line-index per line and >hosts< lines */
  std::vector<int> istartpos, startpos;
  std::vector<double> istarttimes, starttimes;
  if (args_info->traces_start_given) {
    std::ifstream startfile(args_info->traces_start_arg, std::ios::in);
    if (!startfile.is_open()) {
      std::cout << "couldn't open file with start-times - starting with zero"
                << std::endl;
      for (int i = 0; i < hosts; i++) {
        startpos.push_back(0);
        starttimes.push_back(0);
      }
    } else {
      for (int i = 0; i < hosts; i++) {
        char buffer[MAX_STRLEN];
        // class conv_line conv;
        startfile.getline(buffer, MAX_STRLEN);
        // boost::cmatch m;
        // static const boost::regex e_nr("^([\\d]+) (.+)$");
        // if(regex_match(buffer, m, e_nr)) {
        int line;         // conv.read_string(m,1,&line);
        double starttime; // conv.read_string(m,2,&starttime);
        sscanf(buffer, "%i %lf", &line, &starttime);
        startpos.push_back(line);
        starttimes.push_back(starttime);
        //}
      }
      // nope, no better error ...
      if ((int)starttimes.size() < hosts) {
        std::cout << "input file format wrong - exiting" << std::endl;
        return;
      }
    }
    // std::cout << " hosts " << hosts << " " << starttimes.size() << "\n";
    assert(hosts == (int)starttimes.size());
    assert(hosts == (int)startpos.size());
  }

  istartpos = startpos;
  istarttimes = starttimes;
  // loop over extrapolation parameter

  for (int extrhost = 0; extrhost < extrhosts; ++extrhost) {
    std::cout << "extrapolation round " << extrhost << "\n";
    // restore environment as at the beginning
    host = -1;
    startpos = istartpos;
    starttimes = istarttimes;

    while (1) {
      char buffer[MAX_STRLEN];
      change_zero_to_host(args_info->traces_arg, buffer, ++host);
      assert(strlen(buffer) < MAX_STRLEN);

      TraceReader trcrd(buffer);
      if (!trcrd.is_open())
        break;

      if (print)
        std::cout << "# parsing: " << buffer << std::endl;

      double tracestart; // MPI start time of the trace
      LocOp curlocop(&goal, args_info->timemult_arg, args_info->cpu_arg);

      // the map of all open LocOp::REQUests - a request in a trace is
      // identified by an integer - this map saves the identifier of the
      // nonblocking operation associated with the LocOp::REQU. in order to make
      // it dependent to the item after the wait{all,some,any} - D'oh, what do
      // we do with wait{any,some}???
      std::map<unsigned long, Goal::t_id> reqs;
      // Persistent requests
      std::map<unsigned long, p_request_t *> preqs;
      // Clears the comm_map before we start parsing each goal file
      // and resets the comm_index
      comm_map.clear();
      comm_index = 0;

      goal.StartRank(host + hosts * extrhost);
      // boost::cmatch m;

      int found_eof = 0; // flag to indicate when to stop
      int nops = 0;
      int lineno = 0;
      char line[MAX_STRLEN];
      htorParser pars;
      htorMatch match;

      while (!found_eof) {
        // read next line
        found_eof = trcrd.getline(line, MAX_STRLEN);
        lineno++;
        // fast forward to line number from startpos if provided
        if ((int)startpos.size() > 0 && lineno == 1) {

          trcrd.seekg(startpos[host]);

          curlocop.start = starttimes[host];
          if (print)
            std::cout << "rank " << host << " starting at line " << lineno
                      << " with time " << std::setprecision(30)
                      << curlocop.start << "\n";
        }

        /**** start */
        if (strstr(line, "# Init clockdiff: ") != NULL) {
          sscanf(line, "# Init clockdiff: %lf", &tracestart);
          // std::cout << "start: " << tracestart << " at line " << lineno <<
          // "\n";
        }

        // this is just done to speed things up - hash MPI function name
        // once at the beginning and only search function with hash :)
        int funchash = pars.hashFunc(line, pars.hashlength);

        /**** Init */
        static const htorMatcher e_init(&pars, "MPI_Init");
        static const htorMatcher e_init_thread(&pars, "MPI_Init_thread");
        if (pars.match(&e_init, funchash, line, &match)) {
          double inittime;
          match.get(4, &inittime);
          tracestart = inittime;
          if (print)
            std::cout << "starttime host " << host << " = " << tracestart
                      << std::endl;

          assert(curlocop.start == 0);
          curlocop.start = tracestart;
          curlocop.prev = make_vector(std::make_pair(Goal::NO_ID, LocOp::REQU));
          goto endloop;
        }
        /************ Init thread ************/
        if (pars.match(&e_init_thread, funchash, line, &match)) {
          double inittime;
          match.get(6, &inittime);
          tracestart = inittime;
          if (print)
            std::cout << "starttime host " << host << " = " << tracestart
                      << std::endl;

          assert(curlocop.start == 0);
          curlocop.start = tracestart;
          curlocop.prev = make_vector(std::make_pair(Goal::NO_ID, LocOp::REQU));
          goto endloop;
        }


        static const htorMatcher e_rank(&pars, "MPI_Comm_rank");
        static const htorMatcher e_size(&pars, "MPI_Comm_size");
        static const htorMatcher e_getcount(&pars, "MPI_Get_count");
        static const htorMatcher e_probe(&pars, "MPI_Iprobe");
        static const htorMatcher e_testall(&pars, "MPI_Testall");
        static const htorMatcher e_test(&pars, "MPI_Test");
        if (pars.match(&e_rank, funchash, line, &match) ||
            pars.match(&e_probe, funchash, line, &match) ||
            pars.match(&e_testall, funchash, line, &match) ||
            pars.match(&e_test, funchash, line, &match) ||
            pars.match(&e_getcount, funchash, line, &match) ||
            pars.match(&e_size, funchash, line, &match))
          goto endloop;

        static const htorMatcher e_irecv(&pars, "MPI_Irecv");
        static const htorMatcher e_recv(&pars, "MPI_Recv");
        static const htorMatcher e_isend(&pars, "MPI_Isend");
        static const htorMatcher e_send(&pars, "MPI_Send");
        static const htorMatcher e_rsend(&pars, "MPI_Rsend");
        static const htorMatcher e_sendrecv(&pars, "MPI_Sendrecv");
        static const htorMatcher e_wait(&pars, "MPI_Wait");
        static const htorMatcher e_waitall(&pars, "MPI_Waitall");
        // Persistent requests
        static const htorMatcher e_send_init(&pars, "MPI_Send_init");
        static const htorMatcher e_recv_init(&pars, "MPI_Recv_init");
        static const htorMatcher e_start(&pars, "MPI_Start");
        static const htorMatcher e_request_free(&pars, "MPI_Request_free");
        if (args_info->traces_nop2p_given) {
          if (pars.match(&e_irecv, funchash, line, &match) ||
              pars.match(&e_isend, funchash, line, &match) ||
              pars.match(&e_recv, funchash, line, &match) ||
              pars.match(&e_send, funchash, line, &match) ||
              pars.match(&e_rsend, funchash, line, &match) ||
              pars.match(&e_sendrecv, funchash, line, &match) ||
              pars.match(&e_wait, funchash, line, &match) ||
              pars.match(&e_waitall, funchash, line, &match) ||
              // Persistent requests
              pars.match(&e_send_init, funchash, line, &match) ||
              pars.match(&e_recv_init, funchash, line, &match) ||
              pars.match(&e_start, funchash, line, &match) ||
              pars.match(&e_request_free, funchash, line, &match))
            goto endloop;
        } else {
          // MPI_Recv( void *buf, int count, MPI_Datatype datatype, int source,
          // int tag, MPI_Comm comm, MPI_Status *status)
          // MPI_Recv : 1237666053021692.000000 : 11287888 : 4500 : 11,8,8 : 2 :
          // 24000 : 0,0,4 : 140737488340112 : 1237666053022005.000000
          if (pars.match(&e_recv, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int dest;
            match.get(7, &dest);
            int comm;
            match.get(9, &comm);
            double tend;
            match.get(13, &tend);

            if (print)
              std::cout << " recv from " << dest << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << std::endl;

            if (print)
              goal.Comment("Recv begin");
            goal.SetTag(MAKE_TAG(comm, tag));
            if (dest != mpi_any_source /* MPI_ANY_SOURCE */)
              dest += hosts * extrhost;
            Goal::t_id id = goal.Recv(size * count, dest);
            if (print)
              goal.Comment("Recv end");

            curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev = make_vector(std::make_pair(id, LocOp::REQU));

            // nops++; only count colls here
            goto endloop;
          }

          // MPI_Irecv( void *buf, int count, MPI_Datatype datatype, int source,
          // int tag, MPI_Comm comm, MPI_Request *request ) MPI_Irecv :
          // 1225038833071959.000000 : 14744192 : 36864 : 13,1,1 : -1 : 76 :
          // 5641280,0,4 : 14118576 : 1225038833071964.000000
          if (pars.match(&e_irecv, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int dest;
            match.get(7, &dest);
            int comm;
            match.get(9, &comm);
            unsigned long req;
            match.get(12, &req);
            double tend;
            match.get(13, &tend);

            if (print)
              std::cout << " irecv from " << dest << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << " req " << req << std::endl;

            if (print)
              goal.Comment("Irecv begin");
            goal.SetTag(MAKE_TAG(comm, tag));
            if (dest != mpi_any_source /* MPI_ANY_SOURCE */)
              dest += hosts * extrhost;
            Goal::t_id id = goal.Recv(size * count, dest);
            reqs[req] = id;
            if (print)
              goal.Comment("Irecv end");

            curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev = make_vector(std::make_pair(id, LocOp::IREQU));

            // nops++; only count colls here
            goto endloop;
          }

          // MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest,
          // int tag,  MPI_Comm comm ) MPI_Send : 1237666053149864.000000 :
          // 11251872 : 4500 : 11,8,8 : 1 : 7000 : 0,0,4 :
          // 1237666053150082.000000
          if (pars.match(&e_send, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int dest;
            match.get(7, &dest);
            double tend;
            match.get(12, &tend);
            int comm;
            match.get(9, &comm);

            if (print)
              std::cout << " send to " << dest << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << std::endl;
            if (print)
              goal.Comment("Send begin");
            goal.SetTag(MAKE_TAG(comm, tag));
            Goal::t_id id = goal.Send(size * count, dest + hosts * extrhost);
            if (print)
              goal.Comment("Send end");

            curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev = make_vector(std::make_pair(id, LocOp::REQU));

            // nops++; only count colls here
            goto endloop;
          }

          // MPI_Rsend( void *buf, int count, MPI_Datatype datatype, int dest,
          // int tag,  MPI_Comm comm ) MPI_Rsend : 1237844044868539.000000 :
          // 332631168 : 24955 : 10,8,8 : 0 : 0 : 7152208,3,4 :
          // 1237844044868842.000000
          if (pars.match(&e_rsend, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int dest;
            match.get(7, &dest);
            double tend;
            match.get(12, &tend);
            int comm;
            match.get(9, &comm);

            if (print)
              std::cout << " rsend to " << dest << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << std::endl;

            if (print)
              goal.Comment("Rsend begin");
            goal.SetTag(MAKE_TAG(comm, tag));
            Goal::t_id id = goal.Send(size * count, dest + hosts * extrhost);
            if (print)
              goal.Comment("Rsend end");

            curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev = make_vector(std::make_pair(id, LocOp::REQU));

            // nops++; only count colls here
            goto endloop;
          }

          // MPI_Isend( void *buf, int count, MPI_Datatype datatype, int dest,
          // int tag,  MPI_Comm comm, MPI_Request *request ) MPI_Isend :
          // 1225038833277018.000000 : 14781072 : 36864 : 13,1,1 : 1 : 204 :
          // 5641280,0,4 : 7104000 : 1225038833277029.000000
          if (pars.match(&e_isend, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int comm;
            match.get(9, &comm);
            int dest;
            match.get(7, &dest);
            unsigned long req;
            match.get(12, &req);
            double tend;
            match.get(13, &tend);

            if (print)
              std::cout << " isend to " << dest << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << " req " << req << std::endl;

            if (print)
              goal.Comment("Isend begin");
            goal.SetTag(MAKE_TAG(comm, tag));
            Goal::t_id id = goal.Send(size * count, dest + hosts * extrhost);
            reqs[req] = id;
            if (print)
              goal.Comment("Isend end");

            curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev = make_vector(std::make_pair(id, LocOp::IREQU));

            // nops++; only count colls here
            goto endloop;
          }

          /**** Wait */
          // MPI_Wait ( MPI_Request  *request, MPI_Status *status)
          // MPI_Wait : 1225038833273694.000000 : 7104000 : 140734714680384 :
          // 1225038833273749.000000
          if (pars.match(&e_wait, funchash, line, &match)) {
            unsigned long req;
            match.get(2, &req);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(4, &tend);

            // if we cannot find the request, i.e., because its a wait on a
            // MPI_REQUEST_NULL, don't do anything

            if (reqs.find(req) != reqs.end()) {

              // std::cout << line << std::endl;
              if (print)
                std::cout << " wait "
                          << " time " << tend - tstart << " req: " << req
                          << std::endl;
              if (print)
                goal.Comment("wait");
              Goal::t_id id = goal.Exec("wait", 0);
              try {
                Goal::t_id req_id = reqs.at(req);
                // curlocop.prev.push_back(std::make_pair(req_id,LocOp::REQU));
                goal.Requires(id, req_id);
              } catch (std::out_of_range const &) {
                // MPI_Wait will just set the handle to MPI_REQUEST_NULL, its
                // perfectly legal to call again, actually liballprof does not
                // have sufficient info in this case
                //              std::cerr << "request " << req << " not found -
                //              there is something wrong with the trace!" <<
                //              std::endl; return;
              }
              reqs.erase(req);

              curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
              curlocop.NextOp(tstart, tend);
              curlocop.prev = make_vector(std::make_pair(id, LocOp::REQU));
            }
            goto endloop;
          }

          /**** Waitall */
          // MPI_Waitall( int count, MPI_Request array_of_requests[], MPI_Status
          // array_of_statuses[] ) MPI_Waitall : 1237759886409182.000000 : 3 :
          // 6166720 : 5839040 : 1237759886411243.000000
          if (pars.match(&e_waitall, funchash, line, &match)) {
            int nreq;
            match.get(2, &nreq);
            unsigned long req;
            match.get(3, &req);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(5, &tend);

            if (print)
              std::cout << " waitall "
                        << " time " << tend - tstart << " req: " << req
                        << " nreqs: " << nreq << std::endl;
            Goal::t_id id = goal.Exec("waitall", 0);
            try {
              for (unsigned long i = req; i < req + nreq * req_size;
                   i += req_size) {
                if (reqs.find(i) != reqs.end()) {
                  if (print)
                    std::cout << " resolving req " << i << std::endl;
                  Goal::t_id req_id = reqs.at(i);
                  goal.Requires(id, req_id);
                }
              }
            } catch (std::out_of_range const &) {
              // this is bs, doing to MPI_Waits on the same req handle will lead
              // to an error in schedgen, but is perfectly legal, since the
              // first wait will turn the request into an req_null
              std::cerr << "request " << req
                        << " not found - there is something wrong with the "
                           "trace! (try adjusting req_size = "
                        << req_size << ")" << std::endl;
              return;
            }
            reqs.erase(req);

            curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev = make_vector(std::make_pair(id, LocOp::REQU));
            goto endloop;
          }

          // int MPI_Sendrecv( void *sendbuf, int sendcount, MPI_Datatype
          // sendtype, int dest, int sendtag, void *recvbuf, int recvcount,
          // MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
          // MPI_Status *status)
          // MPI_Sendrecv : 1237844041868441.000000 : 140735182514476 : 1 :
          // 1,4,4 : 2 : 0 : 140735182514472 : 1 : 1,4,4 : 2 : 0 : 7152208,0,4 :
          // 140735182514432 : 1237844041868457.000000
          if (pars.match(&e_sendrecv, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int scount;
            match.get(3, &scount);
            int ssize;
            match.get(5, &ssize);
            int stag;
            match.get(8, &stag);
            int sdest;
            match.get(7, &sdest);
            int rcount;
            match.get(10, &rcount);
            int rsize;
            match.get(12, &rsize);
            int rsource;
            match.get(14, &rsource);
            int rtag;
            match.get(15, &rtag);
            int comm;
            match.get(16, &comm);
            double tend;
            match.get(20, &tend);

            if (print)
              std::cout << " sendrecv - send to " << sdest
                        << " size: " << ssize * scount << " tag: " << stag
                        << "; recv from " << rsource
                        << " size: " << rsize * rcount << " tag: " << rtag
                        << "; time: " << tend - tstart << std::endl;

            if (print)
              goal.Comment("Sendrecv begin");
            goal.SetTag(MAKE_TAG(comm, stag));
            Goal::t_id sid =
                goal.Send(ssize * scount, sdest + hosts * extrhost);
            goal.SetTag(MAKE_TAG(comm, stag));
            Goal::t_id rid =
                goal.Recv(rsize * rcount, rsource + hosts * extrhost);
            if (print)
              goal.Comment("Sendrecv end");

            curlocop.next.push_back(std::make_pair(sid, LocOp::REQU));
            curlocop.next.push_back(std::make_pair(rid, LocOp::REQU));
            curlocop.NextOp(tstart, tend);
            curlocop.prev.push_back(std::make_pair(sid, LocOp::REQU));
            curlocop.prev.push_back(std::make_pair(rid, LocOp::REQU));

            // nops++; only count colls here
            goto endloop;
          }
          /**********************************************************************
           * Persistent requests
           **********************************************************************/
          // MPI_Send_init( void *buf, int count, MPI_Datatype datatype, int
          // dest, int tag, MPI_Comm comm, MPI_Request *request )
          // MPI_Send_init:296178103:15263168:147456:13,1,1:3:11:9304208,0,32:15815560:296178106
          if (pars.match(&e_send_init, funchash, line, &match))
          {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int dest;
            match.get(7, &dest);
            int comm;
            match.get(9, &comm);
            unsigned long req;
            match.get(12, &req);
            double tend;
            match.get(13, &tend);

            if (print)
              std::cout << " Created persistent send to " << dest << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << " req " << req << std::endl;
            if (print)
              goal.Comment("Storing persistent send request");
            
            p_request_t *preq = new p_request_t();
            preq->req_id = req;
            preq->is_send = true;
            if (dest != mpi_any_source /* MPI_ANY_SOURCE */)
              dest += hosts * extrhost;
            preq->target = dest;
            preq->tag = tag;
            preq->comm = comm;
            preq->size = size * count;
            // Stores the persistent request in the map
            preqs[req] = preq;
            // This needs to be calibrated
            curlocop.start += (tend - tstart) * REQ_INIT_MULTIPLIER;
            // curlocop.start = tend;
            // nops++; only count colls here
            goto endloop;
          }

          // MPI_Recv_init( void *buf, int count, MPI_Datatype datatype, int
          // source, int tag, MPI_Comm comm, MPI_Request *request )
          // MPI_Recv_init:296178076:15878272:147456:13,1,1:1:11:9304208,0,32:15808776:296178080
          if (pars.match(&e_recv_init, funchash, line, &match))
          {
            double tstart;
            match.get(1, &tstart);
            int size;
            match.get(5, &size);
            int count;
            match.get(3, &count);
            int tag;
            match.get(8, &tag);
            int source;
            match.get(7, &source);
            int comm;
            match.get(9, &comm);
            unsigned long req;
            match.get(12, &req);
            double tend;
            match.get(13, &tend);

            if (print)
              std::cout << " Created persistent recv from " << source << " time: " << tend - tstart
                        << " size: " << size * count << " tag: " << tag
                        << " req " << req << std::endl;
            if (print)
              goal.Comment("Storing persistent recv request");
            
            p_request_t *preq = new p_request_t();
            preq->req_id = req;
            preq->is_send = false;
            if (source != mpi_any_source /* MPI_ANY_SOURCE */)
              source += hosts * extrhost;
            preq->target = source;
            preq->tag = tag;
            preq->comm = comm;
            preq->size = size * count;
            // Stores the persistent request in the map
            preqs[req] = preq;
            // This needs to be calibrated
            curlocop.start += (tend - tstart) * REQ_INIT_MULTIPLIER;
            // curlocop.start = tend;
            // nops++; only count colls here
            goto endloop;
          }

          // MPI_Start( MPI_Request *request )
          // MPI_Start:296178107:15808776:296178107
          if (pars.match(&e_start, funchash, line, &match))
          {
            unsigned long req;
            match.get(2, &req);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(3, &tend);

            if (print)
              std::cout << " start persisten request " << req
                        << " time " << tend - tstart << " req: " << req
                        << std::endl;
            // Makes sure the request is in the map
            if (preqs.find(req) != preqs.end()) {
              p_request_t *preq = preqs.at(req);
              if (print)
                goal.Comment("Persistent request start");
              goal.SetTag(MAKE_TAG(preq->comm, preq->tag));
              Goal::t_id id;
              if (preq->is_send)
              {
                // Send
                id = goal.Send(preq->size, preq->target);
                reqs[req] = id;
              }
              else
              {
                // Recv
                id = goal.Recv(preq->size, preq->target);
                reqs[req] = id;
              }

              if (print)
                goal.Comment("Persistent request end");
              
              curlocop.next = make_vector(std::make_pair(id, LocOp::REQU));
              curlocop.NextOp(tstart, tend);
              curlocop.prev = make_vector(std::make_pair(id, LocOp::IREQU));
            }
            goto endloop;
          }

          // MPI_Request_free( MPI_Request *request )
          // MPI_Request_free:296194441:15815304:296194447
          if (pars.match(&e_request_free, funchash, line, &match))
          {
            unsigned long req;
            match.get(2, &req);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(3, &tend);

            if (print)
              std::cout << " free persisten request " << req
                        << " time " << tend - tstart << " req: " << req
                        << std::endl;
            // Makes sure the request is in the map
            if (preqs.find(req) != preqs.end()) {
              p_request_t *preq = preqs.at(req);
              if (print)
                goal.Comment("Free persistent request");
              // Frees the persistent request
              delete preq;
              preqs.erase(req);
            }
            goto endloop;
          }
        } // p2p communication end

        static const htorMatcher e_barr(&pars, "MPI_Barrier");
        static const htorMatcher e_allred(&pars, "MPI_Allreduce");
        static const htorMatcher e_iallred(&pars, "MPI_Iallreduce");
        static const htorMatcher e_bcast(&pars, "MPI_Bcast");
        static const htorMatcher e_allgather(&pars, "MPI_Allgather");
        static const htorMatcher e_allgatherv(&pars, "MPI_Allgatherv");
        static const htorMatcher e_gatherv(&pars, "MPI_Gatherv");
        static const htorMatcher e_gather(&pars, "MPI_Gather");
        static const htorMatcher e_exscan(&pars, "MPI_Exscan");
        static const htorMatcher e_scatterv(&pars, "MPI_Scatterv");
        static const htorMatcher e_scatter(&pars, "MPI_Scatter");
        static const htorMatcher e_alltoallv(&pars, "MPI_Alltoallv");
        static const htorMatcher e_alltoall(&pars, "MPI_Alltoall");
        static const htorMatcher e_scan(&pars, "MPI_Scan");
        static const htorMatcher e_reduce(&pars, "MPI_Reduce");
        if (args_info->traces_nocolls_given) {
          /* TODO: this is a hack to enable the AMG run to work correctly --
           * this does only increase nops if it would have increased it
           * with colls enabled !! */
          //          if(pars.match(&e_barr, funchash, line, &match) ||
          //             pars.match(&e_allred, funchash, line, &match) ||
          //             pars.match(&e_bcast, funchash, line, &match) ||
          //             pars.match(&e_allgather, funchash, line, &match))
          //             nops++;

          if (pars.match(&e_barr, funchash, line, &match) ||
              pars.match(&e_allred, funchash, line, &match) ||
              pars.match(&e_bcast, funchash, line, &match) ||
              pars.match(&e_allgather, funchash, line, &match) ||
              pars.match(&e_allgatherv, funchash, line, &match) ||
              pars.match(&e_gatherv, funchash, line, &match) ||
              pars.match(&e_gather, funchash, line, &match) ||
              pars.match(&e_exscan, funchash, line, &match) ||
              pars.match(&e_scatterv, funchash, line, &match) ||
              pars.match(&e_scatter, funchash, line, &match) ||
              pars.match(&e_alltoallv, funchash, line, &match) ||
              pars.match(&e_alltoall, funchash, line, &match) ||
              pars.match(&e_scan, funchash, line, &match) ||
              pars.match(&e_reduce, funchash, line, &match))
            goto endloop;
        } else {
          /**********************************************************************
           * Collective Communication start *
           **********************************************************************/

          /**** Barrier */
          // MPI_Barrier:1225632903083161.000000:3,0,32:1225632903083192.000000
          // MPI_Barrier(comm);
          if (pars.match(&e_barr, funchash, line, &match)) {
            double tstart;
            match.get(1, &tstart);
            int p;
            match.get(4, &p);
            int comm;
            match.get(2, &comm);
            double tend;
            match.get(5, &tend);

            // std::cout << line << std::endl;
            if (print)
              std::cout << " barrier " << nops << " (" << p << ") time "
                        << tend - tstart << std::endl;

            if (p != hosts) {
              std::cerr << "[barrier] collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Barrier begin");
            goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
            goal.StartOp();
            create_dissemination_rank(&goal, host + hosts * extrhost,
                                      hosts * extrhosts, 1);
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            finish_coll("barrier", ops, tstart, tend, nbcify, &curlocop, &goal);

            if (print)
              goal.Comment("Barrier end");

            nops++;
            goto endloop;
          }

          /**** Allreduce */
          // MPI_Allreduce(                            sendbuf, recvbuf, count,
          // datatype, op, comm) MPI_Allreduce : 1225632903079389.000000 :
          // 140733419868204 : 140733419868192 : 1 : 2,4,4 : 3 : 5,0,32 :
          // 1225632903079745.000000
          if (pars.match(&e_allred, funchash, line, &match)) {
            int count;
            match.get(4, &count);
            int size;
            match.get(6, &size);
            int comm;
            match.get(9, &comm);
            int p;
            match.get(11, &p);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(12, &tend);

            // std::cout << line << std::endl;
            if (print)
              std::cout << " allreduce " << nops << " (" << p << ") time "
                        << tend - tstart << " size: " << count * size
                        << std::endl;

            if (p != hosts) {
              std::cerr << "[allreduce] collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Allreduce begin");
            goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
            goal.StartOp();
            // create_dissemination_rank(&goal, host + hosts * extrhost,
            //                           hosts * extrhosts, count * size);
            // create_allreduce_ring_rank(&goal, host + hosts * extrhost,
            //                             hosts * extrhosts, count * size);
            create_allreduce_recdoub_rank(&goal, host + hosts * extrhost,
                                          hosts * extrhosts, count * size);
            /* create_allreduce_recdoub_rank(&goal, host + hosts * extrhost, 
                                          hosts * extrhosts, count * size, -1); */
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            finish_coll("allreduce", ops, tstart, tend, nbcify, &curlocop,
                        &goal);

            if (print)
              goal.Comment("Allreduce end");

            nops++;
            goto endloop;
          }

          /**** Iallreduce */
          // MPI_Iallreduce(                            sendbuf, recvbuf, count,
          // datatype, op, comm,               req) MPI_Iallreduce: 59807782 :
          // 139944565018640 : 139944544235536 : 5194816: 8,4,4 : 3 :
          // 94853248035456,0,4 : 140737109594760 : 59813207
          if (pars.match(&e_iallred, funchash, line, &match)) {
            int count;
            match.get(4, &count);
            int size;
            match.get(6, &size);
            int p;
            match.get(11, &p);
            double tstart;
            match.get(1, &tstart);
            double req;
            match.get(12, &req);
            double tend;
            match.get(13, &tend);

            // std::cout << line << std::endl;
            if (print)
              std::cout << " allreduce " << nops << " (" << p << ") time "
                        << tend - tstart << " size: " << count * size
                        << std::endl;

            if (p != hosts) {
              std::cerr << "collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Iallreduce begin");
            goal.SetTag(collsbase + nops);
            goal.StartOp();
            create_dissemination_rank(&goal, host + hosts * extrhost,
                                      hosts * extrhosts, count * size);
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            reqs[req] = finish_coll("iallreduce", ops, tstart, tend, nbcify,
                                    &curlocop, &goal);

            if (print)
              goal.Comment("Iallreduce end");

            nops++;
            goto endloop;
          }

          /**** Bcast */
          // MPI_Bcast(                            void* buffer, int count,
          // MPI_Datatype datatype, int root, MPI_Comm comm ) MPI_Bcast :
          // 1225632902659657.000000 : 140733419888320 : 3 : 21,1,1 : 0 : 0,0,32
          // : 1225632902659726.000000
          if (pars.match(&e_bcast, funchash, line, &match)) {
            int count;
            match.get(3, &count);
            int size;
            match.get(5, &size);
            int root;
            match.get(7, &root);
            int comm;
            match.get(8, &comm);
            int p;
            match.get(10, &p);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(11, &tend);

            if (print)
              std::cout << " bcast " << nops << " (" << p << ") time "
                        << tend - tstart << " size: " << count * size
                        << " root: " << root << std::endl;

            if (p != hosts) {
              std::cerr << "[bcast] collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Bcast begin");
            goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
            goal.StartOp();
            create_binomial_tree_bcast_rank(&goal, root,
                                            host + hosts * extrhost,
                                            hosts * extrhosts, count * size);
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            finish_coll("bcast", ops, tstart, tend, nbcify, &curlocop, &goal);

            if (print)
              goal.Comment("Bcast end");

            nops++;
            goto endloop;
          }

          /**** Allgather */
          // MPI_Allgather ( void *sendbuf, int sendcount, MPI_Datatype
          // sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
          // MPI_Comm comm ) MPI_Allgather : 1237767320811060.000000 :
          // 140734855417288 : 1 : 6204400,8,8 : 383914424 : 1 : 6204400,8,8 :
          // 6201888,0,4 : 1237767320811343.000000
          if (pars.match(&e_allgather, funchash, line, &match)) {
            int count;
            match.get(3, &count);
            int size;
            match.get(5, &size);
            int comm;
            match.get(12, &comm);
            int p;
            match.get(14, &p);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(15, &tend);

            if (print)
              std::cout << " allgather " << nops << " (" << p << ") time "
                        << tend - tstart << " size: " << count * size
                        << std::endl;

            if (p != hosts) {
              std::cerr << "[allgather] collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Allgather begin");
            goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
            goal.StartOp();
            create_dissemination_rank(&goal, host + hosts * extrhost,
                                      hosts * extrhosts, count * size);
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            finish_coll("allgather", ops, tstart, tend, nbcify, &curlocop,
                        &goal);

            if (print)
              goal.Comment("Allgather end");

            nops++;
            goto endloop;
          }

          /**** Scan */
          if (pars.match(&e_scan, funchash, line, &match)) {
            std::cerr << "scan not implemented yet\n";
            goto endloop;
          }

          /**** Reduce */
          // MPI_Reduce(                            void* sbuf, void* rbuf, int
          // count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm )
          // MPI_Reduce : 1237759889225636.000000 : 586555776 : 6756576 :   1960
          // :     11,8,46909632806920 :  3 :        0 :       0,0,4 :
          // 1237759889225839.000000
          if (pars.match(&e_reduce, funchash, line, &match)) {
            int count;
            match.get(4, &count);
            int size;
            match.get(6, &size);
            int root;
            match.get(9, &root);
            int comm;
            match.get(10, &comm);
            int p;
            match.get(12, &p);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(13, &tend);

            if (print)
              std::cout << " reduce " << nops << " (" << p << ") time "
                        << tend - tstart << " size: " << count * size
                        << " root: " << root << std::endl;

            if (p != hosts) {
              std::cerr << "[reduce] collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Reduce begin");
            goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
            goal.StartOp();
            create_binomial_tree_reduce_rank(&goal, root,
                                             host + hosts * extrhost,
                                             hosts * extrhosts, count * size);
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            finish_coll("reduce", ops, tstart, tend, nbcify, &curlocop, &goal);

            if (print)
              goal.Comment("Reduce end");

            nops++;
            goto endloop;
          }

          /**** Alltoall */
          // MPI_Alltoall(                     void *sendbuf, int sendcount,
          // MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype
          // recvtype, MPI_Comm comm ) MPI_Alltoall : 1300050894308309 :
          // 32060384 :     1            : 10,8,8               : 32060432     :
          // 1         :  10,8,8               : 1140850688,0,4  :
          // 1300050894309187
          if (pars.match(&e_alltoall, funchash, line, &match)) {
            int count;
            match.get(3, &count);
            int size;
            match.get(5, &size);
            int comm;
            match.get(12, &comm);
            int p;
            match.get(14, &p);
            double tstart;
            match.get(1, &tstart);
            double tend;
            match.get(15, &tend);

            if (print)
              std::cout << " alltoall " << nops << " (" << p << ") time "
                        << tend - tstart << " size: " << count * size
                        << std::endl;

            if (p != hosts) {
              std::cerr << "[alltoall] collective on subcommunicator p=" << p
                        << ", hosts=" << hosts << " - ignoring\n";
              continue;
            }

            if (print)
              goal.Comment("Alltoall begin");
            goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
            goal.StartOp();
            create_linear_alltoall_rank(&goal, host + hosts * extrhost,
                                        hosts * extrhosts, count * size);
            std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();

            finish_coll("alltoall", ops, tstart, tend, nbcify, &curlocop,
                        &goal);

            if (print)
              goal.Comment("Alltoall end");

            nops++;
            goto endloop;
          }

          /**** Alltoallv */
          if (pars.match(&e_alltoallv, funchash, line, &match)) {
            std::cerr << "alltoallv not implemented yet\n";
            goto endloop;
          }

          /**** Scatter */
          if (pars.match(&e_scatter, funchash, line, &match)) {
            std::cerr << "scatter not implemented yet\n";
            goto endloop;
          }

          /**** Scatterv */
          if (pars.match(&e_scatterv, funchash, line, &match)) {
            std::cerr << "scatterv not implemented yet\n";
            goto endloop;
          }

          /**** Exscan */
          if (pars.match(&e_exscan, funchash, line, &match)) {
            std::cerr << "exscan not implemented yet\nline was: " << line
                      << "\n";
            goto endloop;
          }

          /**** Gather */
          if (pars.match(&e_gather, funchash, line, &match)) {
            std::cerr << "gather not implemented yet\nline was: " << line
                      << "\n";
            goto endloop;
          }

          /**** Gatherv */
          if (pars.match(&e_gatherv, funchash, line, &match)) {
            std::cerr << "gatherv not implemented yet\n";
            goto endloop;
          }

          /**** Allgatherv */
          if (pars.match(&e_allgatherv, funchash, line, &match)) {
            std::cerr << "allgatherv not implemented yet\n";
            goto endloop;
          }
        }

        /**** Finalize */
        static const htorMatcher e_finalize(&pars, "MPI_Finalize");
        if (pars.match(&e_finalize, funchash, line, &match)) {
          double fintime;
          match.get(1, &fintime);

          std::cout << "finalize (virtual) host " << host + hosts * extrhost
                    << " = " << fintime << " (" << (fintime - tracestart) / 1e6
                    << ")" << std::endl;

          curlocop.next = make_vector(std::make_pair(Goal::NO_ID, LocOp::REQU));
          curlocop.NextOp(fintime, fintime);
          goto endloop;
        }

        /* catch everything that wasn't matched! */
        if (print)
          printf("NOT MATCHED: [%s] (good if empty)\n", line);

      endloop:

        /* check if we should exit */
        if (args_info->traces_nops_arg && nops == args_info->traces_nops_arg) {
          if (print)
            std::cout << "processed " << nops
                      << " operations - proceeding to next host\n";
          if (startpos.size()) {
            startpos[host] = trcrd.tellg();
            starttimes[host] = curlocop.start;
          }
          found_eof = 1;
        }
      }
      goal.EndRank();
    }
  }
  goal.Write();

  /* write end times to file */
  if (args_info->traces_start_given) {
    std::fstream startfile(args_info->traces_start_arg, std::ios::out);
    if (!startfile.is_open()) {
      std::cout << "couldn't open file to dump start-times - aborting"
                << std::endl;
      return;
    } else {
      for (int i = 0; i < hosts; i++) {
        startfile << startpos[i] << " " << std::setprecision(30)
                  << starttimes[i] << std::endl;
      }
    }
  }

  std::cout << "found " << host << " trace files" << std::endl;
}
