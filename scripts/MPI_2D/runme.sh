#!/bin/bash

# runme.sh is a shell script for executing GPU MPI application
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh
#------------------------------------------------------------------

# load the reqired modules:
module load cuda
module load openmpi/gcc54-200c

mpirun=$(which mpirun)

# compile the code for Titan X
nvcc -arch=sm_52 --compiler-bindir mpic++ --compiler-options -O3 MPI_Wave_2D_v3.cu

# execute it
# run_cmd="-np 64 -rf gpu_rankfile_64 --mca btl_openib_if_include mlx4_0,mlx4_1 --mca btl_openib_ignore_locality 1 --mca btl_base_verbose 1 a.out"
run_cmd="-np 16 -rf gpu_rankfile_64 --mca btl_openib_if_include mlx4_0,mlx4_1 --mca btl_openib_ignore_locality 1 a.out"

echo $mpirun $run_cmd

$mpirun $run_cmd
