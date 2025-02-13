#/bin/bash

MPICC=mpicc
CC=mpicc
CXX=mpicxx
F77=mpif77

# Building liballprof based on this PR
# https://github.com/spcl/LogGOPSim/pull/2/commits/7cb06f57ebbe0af9b07473c5c415ed6395a552de9

# Compile liballprof
echo "[INFO] Compiling liballprof ..."
./configure MPICC=${MPICC} CC=${CC} CXX=${MPICXX} F77=${F77}
if [ $? -ne 0 ]
then
    # If configure not successful
    echo "[ERROR] 'configure' failed"
    exit 1
fi

# Clean up
make clean

# Add '-fPIC'
echo "[INFO] Inserting '-fPIC' into Makefile"
sed -i 's/CFLAGS = -g -O2 -Wno-missing-prototypes/CFLAGS = -g -O2 -Wno-missing-prototypes -fPIC/' Makefile


if [ $? -ne 0 ]
then
    # If search & replace fails
    echo "[ERROR] Failed to insert '-fPIC' into Makefile"
    exit 1
fi

make -j8


if [ $? -ne 0 ]
then
    # If make fails
    echo "[ERROR] Make failed"
    exit 1
fi

echo "[INFO] Compiling shared libraries for C and Fortran"

cd .libs && \
${MPICC} -shared -o liballprof.so ../mpipclog.o ../sync.o -L$(pwd) -lclog && \
${MPICC} -shared -o liballprof_f77.so ../mpipf77log.o ../sync.o -L$(pwd) -lf77log

if [ $? -ne 0 ]
then
    echo "[ERROR] Compiling shared library failed"
    exit 1
fi

if test -f "liballprof.so" && test -f "liballprof_f77.so"
then
    echo "[INFO] Set up liballprof: SUCCESS"
    cd ..
    # Setup timer lib using makefile.measure-time
    echo "[INFO] Set up timer lib ..."
    CC=${MPICC} make -f makefile.measure-time clean && make -f makefile.measure-time all
    if [ $? -ne 0 ]
    then
        echo "[ERROR] make -f makefile.measure-time all failed"
        exit 1
    fi
    echo "[INFO] Set up timer lib: SUCCESS"

    export LIBALLPROF_C=$(pwd)/liballprof/.libs/liballprof.so
    export LIBALLPROF_F77=$(pwd)/liballprof/.libs/liballprof_f77.so
    export LIBTIMER_C=$(pwd)/liballprof/mpi_timer.so
    export LIBTIMER_F77=$(pwd)/liballprof/mpi_timer_f77.so
else
    echo "[ERROR] liballprof.so or liballprof_f77.so was not created"
    echo "[INFO] Set up liballprof: FAILED"
    exit 1
fi