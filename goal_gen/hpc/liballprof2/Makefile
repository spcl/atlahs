CC=mpicc
CFLAGS=-I/cluster/home/sishen/miniconda3/include -O3 -g
LD_FLAGS=-L/cluster/home/sishen/miniconda3/lib
LIBS=-lunwind -lc
C_SRC=mpi_c_wrapper.c
F77_SRC=mpi_f_wrapper.c

all: c f77

c: mpi_c_wrapper.o
	${CC} -shared -fPIC -Wl,-soname,liballprof.so.1 ${LD_FLAGS} -o liballprof.so $< ${LIBS}

f77: mpi_f77_wrapper.o
	${CC} -shared -fPIC -Wl,-soname,liballprof_f77.so.1 ${LD_FLAGS} -o liballprof_f77.so $< ${LIBS}

mpi_c_wrapper.o:
	${CC} -fPIC ${CFLAGS} -o $@ -c ${C_SRC}

mpi_f77_wrapper.o:
	${CC} -fPIC ${CFLAGS} -o $@ -c ${F77_SRC}

clean:
	rm -f mpi_c_wrapper.o
	rm -f mpi_f77_wrapper.o
	rm -f liballprof.so
	rm -f liballprof_f77.so
