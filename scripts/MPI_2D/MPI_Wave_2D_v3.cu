// Wave 2D GPU Cuda aware MPI
// nvcc -arch=sm_52 --compiler-bindir mpic++ --compiler-options -O3 MPI_Wave_2D_v3.cu
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"
#define NDIMS  2

#define USE_SINGLE_PRECISION    /* Comment this line using "//" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT      float
#define PRECIS   4
#else
#define DAT      double
#define PRECIS   8
#endif
#define GPU_ID   3
#define OVERLENGTH_X  1
#define OVERLENGTH_Y  1
#define zeros(A,nx,ny)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny))*sizeof(DAT)); \
                        for(i=0; i < ((nx)*(ny)); i++){ A##_h[i]=(DAT)0.0; }              \
                        cudaMalloc(&A##_d      ,((nx)*(ny))*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,((nx)*(ny))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)     free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny) cudaMemcpy( A##_h,A##_d,((nx)*(ny))*sizeof(DAT),cudaMemcpyDeviceToHost);
// --------------------------------------------------------------------- //
// Physics
const DAT Lx   = 40.0;
const DAT Ly   = 10.0;
const DAT k    = 1.0;
const DAT rho  = 1.0;
// Numerics
#define BLOCK_X  32
#define BLOCK_Y  32
#define GRID_X   24
#define GRID_Y   24
#define DIMS_X   8
#define DIMS_Y   2
const int nx = BLOCK_X*GRID_X - OVERLENGTH_X;
const int ny = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
const int nt = 2200;
// Preprocessing
DAT    dx, dy;
size_t Nix, Niy, Niz;
// GPU
#include "geocomp_unil_mpi2D.h"

// Computing physics kernels /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void init(DAT* x, DAT* y, DAT* P, int* coords, const DAT Lx, const DAT Ly, DAT dx, DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y

    if (iy<ny && ix<nx){ x[ix + iy*nx] = (DAT)(coords[0]*(nx-2) + ix)*dx - (DAT)0.5*Lx; }
    if (iy<ny && ix<nx){ y[ix + iy*nx] = (DAT)(coords[1]*(ny-2) + iy)*dy - (DAT)0.5*Ly; }
    if (iy<ny && ix<nx){ P[ix + iy*nx] = exp(-(x[ix + iy*nx]*x[ix + iy*nx]) -(y[ix + iy*nx]*y[ix + iy*nx])); }
}
__global__ void compute_V(DAT* Vx, DAT* Vy, DAT* P, DAT dt, const DAT rho, DAT dx, DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y

    if (iy<ny && ix>0 && ix<nx){
        Vx[ix + iy*(nx+1)] = Vx[ix + iy*(nx+1)] - (dt/dx/rho)*(P[ix + iy*nx]-P[ix-1 +  iy   *nx]); }
    if (iy>0 && iy<ny && ix<nx){
        Vy[ix + iy*(nx  )] = Vy[ix + iy*(nx  )] - (dt/dy/rho)*(P[ix + iy*nx]-P[ix   + (iy-1)*nx]); }
}
__global__ void compute_P(DAT* Vx, DAT* Vy, DAT* P, DAT dt, const DAT k, DAT dx, DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    
    if (iy<ny && ix<nx){
        P[ix + iy*nx] = P[ix + iy*nx] - dt*k*(((DAT)1.0/dx)*(Vx[(ix+1) + iy    *(nx+1)]-Vx[ix + iy*(nx+1)]) 
                                            + ((DAT)1.0/dy)*(Vy[ ix    + (iy+1)* nx   ]-Vy[ix + iy* nx   ]) ); }
}

int main(int argc, char *argv[]){
    int i, it;
    set_up_grid();
    set_up_parallelisation();
    // Initial arrays
    zeros(x  ,nx  ,ny  );
    zeros(y  ,nx  ,ny  );
    zeros(P  ,nx  ,ny  );
    zeros(Vx ,nx+1,ny  );
    zeros(Vy ,nx  ,ny+1);
    // MPI sides    
    init_sides(Vx ,nx+1,ny  );
    init_sides(Vy ,nx  ,ny+1);
    // Preprocessing
    Nix  = ((nx-2)*dims[0])+2;
    Niy  = ((ny-2)*dims[1])+2;
    dx   = Lx/((DAT)Nix-(DAT)1.0);  // Global dx, dy
    dy   = Ly/((DAT)Niy-(DAT)1.0);
    DAT dt = min(dx,dy)/sqrt(k/rho)/4.1;
    // Initial conditions
    init<<<grid,block>>>(x_d, y_d, P_d, coords_d, Lx, Ly, dx, dy, nx, ny);    cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        if (it==3){ tic(); }
        if (it>0){
        compute_P<<<grid,block>>>(Vx_d, Vy_d, P_d, dt, k,   dx, dy, nx, ny);  cudaDeviceSynchronize(); }
        compute_V<<<grid,block>>>(Vx_d, Vy_d, P_d, dt, rho, dx, dy, nx, ny);  cudaDeviceSynchronize();
        update_sides(Vx,nx+1,ny  );
        update_sides(Vy,nx  ,ny+1);
    }//it
    tim("Performance", Nix*Niy*(nt-3)*6*PRECIS/(1e9)); // timer test
    // printf("Process %d used GPU with id %d.\n",me,gpu_id);
    free_all(x );
    free_all(y );
    free_all(P );
    free_all(Vx);
    free_all(Vy);
    // MPI
    free_sides(Vx);
    free_sides(Vy);
    
    clean_cuda();
    MPI_Finalize();
    return 0;
}
