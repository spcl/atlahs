/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include "schedgen.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string.h>
#include <unistd.h>

//#define HAVE_BOOST_IO

#ifdef HAVE_BOOST_IO
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/char_traits.hpp>
#endif

class TraceReader {
  private:
  std::ifstream trace;
  enum {BZ2, NORM} type;

#ifdef HAVE_BOOST_IO
  boost::iostreams::filtering_streambuf<boost::iostreams::input> inbz2;
#endif
  
  public:
  TraceReader(std::string fname) {
    trace.open(fname.c_str(),std::ios::in);

    //boost::cmatch m;
    //static const boost::regex e(".*\\.bz2$");
    //if(regex_match(fname.c_str(), m, e)) type = BZ2;
    if(NULL!=strstr(fname.c_str(), ".bz2")) {
#ifdef HAVE_BOOST_IO
      type = BZ2;
#else
      std::cerr << "bz2 not supported (anymore)\n";
      _exit(10);
#endif
    } else type=NORM;

#ifdef HAVE_BOOST_IO
    if(type == BZ2) {
      inbz2.push(boost::iostreams::bzip2_decompressor());
      inbz2.push(trace);
    }
#endif
  }

  bool is_open() {
    return trace.is_open();
  }

  std::streampos tellg() {
    return trace.tellg();
  }

  void seekg(std::streampos pos) {
    trace.seekg(pos);
  }

  bool getline(char* s, int n) {
    bool eof=0;
    if(type == BZ2) {
#ifdef HAVE_BOOST_IO
      int pos = 0;
      while(1) {
        std::string line;
        char z = boost::iostreams::get(inbz2);
        if(z == '\n') break;
        if(z == EOF) { eof=1; break; }
        s[pos++] = z;
      }
      s[pos]='\0';
#endif
    } else {
      trace.getline(s,n);
      eof = trace.eof();
    }
    //std::cout << "getline " << s << "\n";
    return eof;
  }
};
