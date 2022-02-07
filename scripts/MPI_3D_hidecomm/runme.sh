#!/bin/bash

# runme.sh is a shell script for executing GPU MPI application
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh
#------------------------------------------------------------------

# load the reqired modules:
module load cuda/8.0
module load openmpi/gcc54-200c

mpirun=$(which mpirun)
nvprof=$(which nvprof)

# How many mpi procs should run
if [ $# -lt 1 ]; then
    echo $0: usage: runme.sh nprocs
    exit 1
fi

nprocs=$1

# compile the code for Titan X
nvcc -arch=sm_52 --compiler-bindir mpic++ --compiler-options -O3 MPI_Wave_3D_hidecomm.cu -DD_x=$2 -DD_y=$3 -DD_z=$4

# run
run_cmd="-np $nprocs -rf gpu_rankfile_128 --mca btl_openib_if_include mlx4_0,mlx4_1 --mca btl_openib_ignore_locality 1 a.out"

echo $mpirun $run_cmd

$mpirun $run_cmd
