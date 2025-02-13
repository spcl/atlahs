/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include <stdint.h>

// This class stores the information of a single element of an address list
// These things consist of three entries: The type, which can be IN=1 or OUT=2,
// indicated by a '<' or '>' in the schedule. Then there is the actual address
// in memory which is a simple integer in our language. And last, there is the
// size of data referenced by this address which is an integer and denotes the
// size in bytes.

typedef uint64_t btime_t;

class buffer_element {
	public:
		int type; // IN=1, OUT=2
		int addr; // address where to read/write
		btime_t size; // size of data to read/write in bytes

		buffer_element() : type(0), addr(0), size(0) {};
		buffer_element(const buffer_element &elem) : type(elem.type), addr(elem.addr), size(elem.size) {};
		buffer_element(int t, int a, btime_t s) : type(t), addr(a), size(s) {};
		buffer_element& operator=(const buffer_element &elem) {
      type = elem.type;
      addr = elem.addr;
      size = elem.size;
      return *this;
    };
		
};

