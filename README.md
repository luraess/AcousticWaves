# Acoustic Waves

Explicit FDTD parallel GPU acoustic wave solver

## Description

This repository contains 2D and 3D finite-difference time-domain (FDTD) acoustic wave solvers, targeting parallel GPU hardware using CUDA-aware MPI. The codes are located in the [scripts](scripts/) folder, which contains:
- [2D CUDA-aware MPI codes](scripts/MPI_2D/)
- [3D CUDA-aware MPI codes](scripts/MPI_3D/)
- [2D CUDA-aware MPI codes](scripts/MPI_2D_hidecomm/) including communication and computation overlap (hidecomm)
- [3D CUDA-aware MPI codes](scripts/MPI_3D_hidecomm/) including communication and computation overlap (hidecomm)
