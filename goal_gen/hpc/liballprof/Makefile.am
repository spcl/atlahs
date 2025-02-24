noinst_HEADERS = sync.h numbers.h mpi_helloworld.c mpi_helloworld.f90 allprof.h wrapper.sh

lib_LTLIBRARIES = libclog.la libf77log.la

libclog_la_SOURCES = mpipclog.c sync.c
libf77log_la_SOURCES = mpipf77log.c sync.c

mpi_helloworldc: mpi_helloworld.c libclog.la
	$(CC) $(LDFLAGS) $< -o $@ -L.libs -lclog                                                                                                                                         
                                                                                                                                                                               
mpi_helloworldf77: mpi_helloworld.f90 libf77log.la
	$(FC) $(LDFLAGS) $< -o $@ -L.libs -lf77log 


# add to the clean target
clean-local:
	rm -f mpi_helloworldc mpi_helloworldf77
