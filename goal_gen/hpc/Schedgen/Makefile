CXXFLAGS=-g -O3 -Wno-deprecated -Wall
CCFLAGS=-g -O3 -g
LDFLAGS=-g -O3 -g #-lboost_iostreams-mt 

force: all

cmdline.c: schedgen.ggo
	gengetopt < $<

%.o: %.cpp *hpp *h
	g++ $(CXXFLAGS) -c $<

%.o: %.c *h
	gcc $(CCFLAGS) -c $<

all: buffer_element.o  cmdline.o  process_trace.o  schedgen.ggo  schedgen.o 
	g++ $(CXXFLAGS) *.o -o schedgen $(LDFLAGS)

clean:
	rm -f *.o
	rm -f cmdline.c cmdline.h
	rm -f schedgen
