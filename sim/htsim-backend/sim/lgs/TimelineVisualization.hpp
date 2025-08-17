#include <string>
#include <math.h>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <vector>

class TimelineVisualization {

	private:
	std::string content;
  bool enable;
	std::string filename;

  void add_ranknum(int numranks) {

    std::stringstream os;
    os << "numranks " << numranks << ";\n";
    this->content.append(os.str());

  }

  void write_events(bool append) {

    std::ofstream myfile;
    if (append) myfile.open(filename.c_str(), std::ios::out | std::ios::app);
    else myfile.open(filename.c_str(), std::ios::out);
    if (myfile.is_open()) {
      myfile << this->content;
      myfile.close();
    }
    else {
      fprintf(stderr, "Unable to open %s\n", filename.c_str());
    }
    
  }



	public:

  TimelineVisualization(gengetopt_args_info *args_info, int p) : enable(enable) {
    enable = args_info->vizfile_given;
    if(!enable) return;

    filename = args_info->vizfile_arg;
    add_ranknum(p);
  }

  ~TimelineVisualization() {
    if(!enable) return;

    write_events(false);
  }
	
  void add_osend(int rank, int start, int end, int cpu, float r=0.0, float g=0.0, float b=1.0) {
    if(!enable) return;
    
    std::stringstream outstream;
    outstream << "osend " << rank << " " << cpu << " " << start << " " << end << " " << r << " " << g << " " << b << ";\n";
    this->content.append(outstream.str());

  }

  void add_orecv(int rank, int start, int end, int cpu, float r=0.0, float g=0.0, float b=1.0) {
    if(!enable) return;
    
    std::stringstream os;
    os << "orecv " << rank << " " << cpu << " " << start << " " << end << " " << r << " " << g << " " << b << ";\n";
    this->content.append(os.str());

  }

  void add_loclop(int rank, int start, int end, int cpu, float r=1.0, float g=0.0, float b=0.0) {
    if(!enable) return;
    
    std::stringstream os;
    os << "loclop " << rank << " " << cpu << " " << start << " " << end << " " << r << " " << g << " " << b << ";\n";
    this->content.append(os.str());

  }
  
  void add_noise(int rank, int start, int end, int cpu, float r=0.0, float g=1.0, float b=0.0) {
    if(!enable) return;
    
    std::stringstream os;
    os << "noise " << rank << " " << cpu << " " << start << " " << end << " " << r << " " << g << " " << b << ";\n";
    this->content.append(os.str());

  }

  void add_transmission(int source, int dest, int starttime, int endtime, int size, int G, float r=0.0, float g=0.0, float b=1.0) {
    if(!enable) return;
    
    std::stringstream os;
    os << "transmission " << source << " " << dest << " " << starttime << " ";
    os << endtime << " " << size << " " << G <<  " " << r << " " << g << " " << b << ";\n";
    this->content.append(os.str());
  }
};
