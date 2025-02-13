CXX=g++
CXXFLAGS= -O0 -g -pedantic -Wno-deprecated -Wall -Wno-long-long -I/usr/include/graphviz
CCFLAGS= -O0 -g 
LDFLAGS= -lcgraph -g

AUTOGEN_SRC= cmdline.c cmdline.h 
LOGGOPSIM_OBJECTS= LogGOPSim.o 
HLPR_OBJECTS= cmdline.o 
ALL_OBJECTS= $(LOGGOPSIM_OBJECTS) $(HLPR_OBJECTS)
BINARY= LogGOPSim

all: $(ALL_OBJECTS) cmdline.h txt2bin
	$(CXX) $(CXXFLAGS) $(ALL_OBJECTS) -o $(BINARY) $(LDFLAGS)

txt2bin:
	re2c -o txt2bin.cpp txt2bin.re
	gengetopt --file-name=cmdline_txt2bin < txt2bin.ggo
	$(CXX) -g -O3 txt2bin.cpp cmdline_txt2bin.c -o txt2bin

cmdline.c: simulator.ggo 
	gengetopt < simulator.ggo

cmdline.h: simulator.ggo
	gengetopt < simulator.ggo

%.o: %.cpp $(AUTOGEN_SRC) *.hpp 
	$(CXX) $(CXXFLAGS) -c $<

%.o: %.c $(AUTOGEN_SRC) *.h
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f $(AUTOGEN_SRC)
	rm -f $(ALL_OBJECTS)
	rm -f $(BINARY)
	rm -f txt2bin.cpp txt2bin bin2txt bin2dot simtest
	rm -f cmdline_txt2bin.*
