#include "cmdline.h"
#include <iostream>
#include <assert.h>
#include <fstream>
#include <vector>
#include "MersenneTwister.h"
#include <limits>

class Noise {
  private:
    int p;
    std::vector<std::pair<uint64_t,uint64_t> > trcnoise; // read NG noise
    uint64_t trctime; // how long is the trace

    std::vector<uint64_t> ranktime; // time in trcnoise for each rank
    std::vector<uint64_t> injected_noise; // counts total injected noise per node 
    static const int max_report=64; // maximum number of nodes to report for

	public:

  Noise(gengetopt_args_info *args_info, int p) : p(p) {

    if(args_info->noise_trace_given) {
      const int size=1024;
      char buffer[size];
      std::ifstream trace;
      trace.open(args_info->noise_trace_arg);
      if(!trace.is_open()) { 
        std::cerr << "couldn't read noise trace file: " << args_info->noise_trace_arg << " - exiting\n";
        throw(10);
      }
      
      bool eof=false;
      int line=0;
      while(!eof) {
        line++;
        trace.getline(buffer,size);

        if(buffer[0] == '#') continue;

        double offset, duration;
        // format: line ::= <start-time>\t<duration> - all times in nanoseconds
        sscanf(buffer, "%lf\t%lf", &offset, &duration);
        
        //std::cout << offset << " " << duration << "\n";
        trcnoise.push_back(std::make_pair((uint64_t)round(offset), (uint64_t)round(duration)));

        eof = trace.eof();
      }
      
      if(((trcnoise.end()-1)->first-trcnoise.begin()->first) > (double)std::numeric_limits<uint64_t>::max()) {
        std::cerr << " the length of the noise-trace ("<<(trcnoise.end()-1)->first-trcnoise.begin()->first<<" ns) is can not be saved in 'uint64_t' (max: "<<(double)std::numeric_limits<uint64_t>::max()<<") - exiting\n";
        throw(11);
      }

      //trctime = ((trcnoise.end()-1)->first-trcnoise.begin()->first);
      trctime = (trcnoise.end()-1)->first;
      std::cout << "Noisegen: read " << trcnoise.size() << " noise events spanning " << trctime/1e9 << "s ";
      if(args_info->noise_cosched_given)
        std::cout << "(coscheduling)\n";
      else
        std::cout << "(independent)\n";

      MTRand mtrand;
      double cosched_starttime = mtrand.rand((double)trctime);
      for(int i=0; i<p; i++) {
        if(args_info->noise_cosched_given) {
          ranktime.push_back((uint64_t)cosched_starttime);
        } else {
          double starttime = mtrand.rand((double)trctime);
          ranktime.push_back((uint64_t)starttime);
          //printf("%i %llu %llu\n", i, (uint64_t)starttime, trctime);
        }
        if (p<=max_report) injected_noise.push_back(0);
      }
    }
  }

  ~Noise() {
    // if we have trace data
    if(trcnoise.size()) {
      // only print noise for small runs
      if (p<=max_report) {
        std::cout << "noise per rank: ";
        for(int i=0; i<p; i++) {
          std::cout << injected_noise[i] << ",";
        }
      }
      std::cout << "\n";
    }
  }
	
	inline btime_t get_noise(int r, btime_t starttime /* in ns */, btime_t endtime /* in ns */) {
    btime_t noise=0;

    // if we have trace data
    if(trcnoise.size()) {
      btime_t oplength = endtime-starttime;
      // start time in trace -- endtime must not be larger than the last
      // entry in trace because the binary search below doesn't work otherwise
      btime_t trcstart = (starttime+ranktime[r])%(trcnoise.back().first-oplength);

      //std::cout << "start: " << starttime << "-" << endtime << " " << noise << "\n";

      unsigned int pos=0;
      btime_t endlastevent=0;
      // trcstart smaller than first elemens has to be a special case!
      if(trcstart > trcnoise[0].first) {
        // do binary search for pos where trcnoise[pos].first is the
        // biggest element that is smaller than trcstart
        unsigned int min=0, max=trcnoise.size()-1;
        do {
          pos=(min+max) / 2;
          if(trcstart > trcnoise[pos].first) {
            min = pos+1;
          } else {
            max = pos-1;
          }
        } while((trcstart != trcnoise[pos].first) && (min < max));

        // the binary search doesn't necessarily find the right interval,
        // however, it brings us close
        while( !( // we loop until we have:
              (trcnoise[pos].first <= trcstart) && // pos is smaller or equal than trcstart
              (trcnoise[pos+1].first > trcstart)   // pos+1 is larger than trcstart
              ) ) {
          if(trcnoise[pos].first > trcstart) pos--;
          else pos++;
        };
        // compute the endtime of the last event
        endlastevent = trcnoise[pos].first+trcnoise[pos].second;
      }

      // if last event reached into starttime - then it influenced me :)
      if(endlastevent>trcstart) {
        noise += endlastevent-trcstart;
      }

      //if(noise > 100000) std::cout << trcnoise[pos].first << " " << trcstart << " " << trcnoise[pos+1].first << "\n";
      
      btime_t end = trcstart+oplength;

      // if we're at the end of samples - wrap around
      if(pos == trcnoise.size()-1) {
        end -= trctime; // adjust end time
        pos = 0; // set position to first 
      }

      // if we reach into next sample - then add the whole time of sample
      while(end > trcnoise[pos+1].first) {
        pos++;
        noise += trcnoise[pos].second;
        /*if(noise > 100000) {
          std::cout << "inner " << trcnoise[pos].first << " pos: " << pos << " start-end: " << trcstart << "-" << end << " end: " << end << " noise: " << noise << " trctime: " << trctime << "\n";
        return 0;}*/
        
        // if we're at the end of samples - wrap around
        if(pos == trcnoise.size()-1) {
          end -= trctime; // adjust end time
          pos = 0; // set position to first 
        }
      }

      // do *NOT* update ranktime because starttime is absolut

      //if (noise > 100000) std::cout << "injected " << noise << " ns noise in " << endtime-starttime << "ns\n";
      if (p<=max_report) injected_noise[r] += noise;
    }
   
    assert(noise >= 0); 
    return noise;
  }
};
