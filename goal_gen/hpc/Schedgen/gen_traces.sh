#!/bin/bash
rm traces/*
GENERATE_ALLTOALL=true
GENERATE_ALLTOALLV=true
GENERATE_ALLREDUCE=true

for COMMSIZE in 16 128 1024 4096
do
    for DATASIZE in 1 16 128 1024 16384
    do
        # Alltoall
        if [ "$GENERATE_ALLTOALL" = true ]; 
        then
            ./schedgen -p linear_alltoall -d ${DATASIZE} -s ${COMMSIZE} -o traces/linear_alltoall_${COMMSIZE}_${DATASIZE}.goal
        fi

        # Alltoallv
        if [ "$GENERATE_ALLTOALLV" = true ]; 
        then
            for SKEW_RATIO in 1 10 100 1000
            do
                if [ $DATASIZE -ge $SKEW_RATIO ];
                then
                    ./schedgen -p linear_alltoallv -d ${DATASIZE} -s ${COMMSIZE} --a2av-skew-ratio ${SKEW_RATIO} -o traces/linear_alltoallv_${COMMSIZE}_${DATASIZE}_${SKEW_RATIO}.goal
                    ./schedgen -p linear_alltoallv -d ${DATASIZE} -s ${COMMSIZE} --a2av-skew-ratio ${SKEW_RATIO} --outcast -o traces/linear_alltoallv_${COMMSIZE}_${DATASIZE}_${SKEW_RATIO}_outcast.goal
                fi
            done
        fi
    done

    for DATASIZE in 1 16 128 1024 16384 131072 1048576
    do
        if [ "$GENERATE_ALLREDUCE" = true ]; 
        then
            if [ $DATASIZE -ge $COMMSIZE ];
            then        
                # Allreduce - Recursive Doubling
                ./schedgen -p allreduce_recdoub -d ${DATASIZE} -s ${COMMSIZE} -o traces/allreduce_recdoub_${COMMSIZE}_${DATASIZE}.goal
                for CMP in 0 1 10 100 1000 10000 100000
                do
                    ./schedgen -p allreduce_recdoub -d ${DATASIZE} -s ${COMMSIZE} --rpl-dep-cmp ${CMP} -o traces/allreduce_recdoub_${COMMSIZE}_${DATASIZE}_${CMP}.goal
                done
                # Allreduce - Ring
                ./schedgen -p allreduce_ring -d ${DATASIZE} -s ${COMMSIZE} -o traces/allreduce_ring_${COMMSIZE}_${DATASIZE}.goal
            fi
        fi
    done
done