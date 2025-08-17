# GOAL Generation for HPC Applications


## Tracing
To trace an MPI application, you have to first compile the `liballprof` library, which is a library that traces the MPI calls with the PMPI interface. Execute the following command to compile the library:
```bash
> cd liballprof
> ./setup.sh # Make sure to change the compilers (e.g., CC, CXX, etc.) to the ones that are available on your system in the script
```
The shared library `liballprof.so` and `liballprof_f77.so` should be found in the `.libs/` directory.

To trace ICON, however, you will have to compile `liballprof2` instead:
```bash
> cd liballprof2
> make
```

When you want to trace an application, simply export the `LD_PRELOAD` environment variable to the path of the `liballprof.so` or `liballprof2.so` library, depending on whether it is a Fortran or C application. For example, to trace the `mpirun` command, you can run the following command:
```bash
> export LD_PRELOAD=/path/to/liballprof.so
> mpirun -n 4 ./a.out
```

By default, the traces will be saved in the `/tmp/` directory. You can change the directory by setting the `HTOR_PMPI_FILE_PREFIX` environment variable. For example, to save the traces in the `traces/` directory, you can run the following command:
```bash
> export HTOR_PMPI_FILE_PREFIX=/path/to/traces/
> export LD_PRELOAD=/path/to/liballprof.so
> mpirun -n 4 ./a.out
```
In an PMPI trace file, each line corresponds to one MPI call. The format of the line is as follows:
```
<function>:<start_time>:<other-args>:<end_time>
```
where `<function>` is the name of the MPI function, `<start_time>` is the start time of the MPI call, `<other-args>` includes all other arguments of the operation, and `<end_time>` is the end time of the function call.


## Converting PMPI Traces to GOAL Files

To convert the raw PMPI traces to GOAL files, you have to use `Schedgen`. To compile `Schedgen`, you can run the following command:
```bash
> cd Schedgen
> make
```

To convert the PMPI traces to GOAL files, you can run the following command:
```bash
> ./schedgen -p trace --traces <path-to-rank-0-trace-file> -o <path-to-goal-file>
```

The `-p trace` option tells `Schedgen` to convert the PMPI traces to GOAL files. The `--traces` option specifies the path to the PMPI trace file for rank 0, the rest of the ranks will be automatically detected. The `-o` option specifies the path to the GOAL file that will be generated.


Note that both `liballprof` and `Schedgen` are automatically compiled when executing the following command in the root directory in the Docker container:
```bash
> docker run -v $(pwd):/workspace atlahs build -t
```