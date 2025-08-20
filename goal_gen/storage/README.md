# GOAL Generation for Distributed Storage Systems

This directory contains the GOAL generation toolchain for distributed storage systems. 

The tracer tool is used to trace the block I/O events of the storage system, and can be found in the `block-io-tracer` directory.

The DirectDriveSim tool is used to convert storage block I/O traces to GOAL files that represent the workload of storage requests specifically for the [Azure Direct Drive](https://www.youtube.com/watch?v=fhFhxSD42WI&ab_channel=SNIAVideo). It can be found in the `DirectDriveSim` directory. It is the work done by [Pasquale Jordan](https://github.com/TrimVis), and all credit goes to him. The documentation of the tool can be found in pdf file `DirectDrive_Documentation.pdf`. Note that Direct Drive serves as a mere example to show that ATLAHS is capable of converting storage block I/O traces to GOAL files. Users can use it as a reference to design and develop GOAL parsers that suit their own storage system.