// Wave 3D GPU Cuda aware MPI
// nvcc -arch=sm_52 --compiler-bindir mpic++ --compiler-options -O3 MPI_Wave_3D_v4.cu
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"
#define NDIMS  3

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
#define OVERLENGTH_Z  1
#define BOUNDARY_WIDTH_X 16
#define BOUNDARY_WIDTH_Y 16
#define BOUNDARY_WIDTH_Z 16

#define zeros(A,nx,ny,nz)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny)*(nz))*sizeof(DAT)); \
                           for(i=0; i < ((nx)*(ny)*(nz)); i++){ A##_h[i]=(DAT)0.0; }              \
                           cudaMalloc(&A##_d      ,((nx)*(ny)*(nz))*sizeof(DAT));                 \
                           cudaMemcpy( A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)        free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny,nz) cudaMemcpy( A##_h,A##_d,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyDeviceToHost);
// --------------------------------------------------------------------- //
// Physics
const DAT Lx   = 10.0;
const DAT Ly   = 10.0;
const DAT Lz   = 10.0;
const DAT k    = 1.0;
const DAT rho  = 1.0;
// Numerics
#define BLOCK_X  32
#define BLOCK_Y  16
#define BLOCK_Z  2
#define GRID_X   2*8
#define GRID_Y   4*8
#define GRID_Z   32*8
#define DIMS_X   D_x
#define DIMS_Y   D_y
#define DIMS_Z   D_z
const int nx = BLOCK_X*GRID_X - OVERLENGTH_X;
const int ny = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
const int nz = BLOCK_Z*GRID_Z - OVERLENGTH_Z;
const int nt = 150;
// Preprocessing
DAT    dx, dy, dz;
size_t Nix, Niy, Niz;
// GPU MPI
#include "geocomp_unil_mpi3D.h"

// Computing physics kernels /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void init(DAT* x, DAT* y, DAT* z, DAT* P, int* coords, const DAT Lx, const DAT Ly, const DAT Lz, DAT dx, DAT dy, DAT dz, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    if (iz<nz && iy<ny && ix<nx){ x[ix + iy*nx + iz*nx*ny] = (DAT)(coords[0]*(nx-2) + ix)*dx - (DAT)0.5*Lx; }
    if (iz<nz && iy<ny && ix<nx){ y[ix + iy*nx + iz*nx*ny] = (DAT)(coords[1]*(ny-2) + iy)*dy - (DAT)0.5*Ly; }
    if (iz<nz && iy<ny && ix<nx){ z[ix + iy*nx + iz*nx*ny] = (DAT)(coords[2]*(nz-2) + iz)*dz - (DAT)0.5*Lz; }
    if (iz<nz && iy<ny && ix<nx){ P[ix + iy*nx + iz*nx*ny] = exp(-(x[ix + iy*nx + iz*nx*ny]*x[ix + iy*nx + iz*nx*ny]) -(y[ix + iy*nx + iz*nx*ny]*y[ix + iy*nx + iz*nx*ny]) -(z[ix + iy*nx + iz*nx*ny]*z[ix + iy*nx + iz*nx*ny])); }
}
__global__ void compute_V(DAT* Vx, DAT* Vy, DAT* Vz, DAT* P, DAT dt, const DAT rho, DAT dx, DAT dy, DAT dz, const int nx, const int ny, const int nz, int istep){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    CommOverlap();

    if (iz<nz && iy<ny && ix>0 && ix<nx){
        Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] = Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] - (dt/dx/rho)*(P[ix + iy*nx + iz*nx*ny]-P[(ix-1) + (iy  )*nx + (iz  )*nx*ny]); }

    if (iz<nz && iy>0 && iy<ny && ix<nx){
        Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)] = Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)] - (dt/dy/rho)*(P[ix + iy*nx + iz*nx*ny]-P[(ix  ) + (iy-1)*nx + (iz  )*nx*ny]); }

    if (iz>0 && iz<nz && iy<ny && ix<nx){
        Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )] = Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )] - (dt/dz/rho)*(P[ix + iy*nx + iz*nx*ny]-P[(ix  ) + (iy  )*nx + (iz-1)*nx*ny]); }
}
__global__ void compute_P(DAT* Vx, DAT* Vy, DAT* Vz, DAT* P, DAT dt, const DAT k, DAT dx, DAT dy, DAT dz, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    if (iz<nz && iy<ny && ix<nx){
        P[ix + iy*nx + iz*nx*ny] = P[ix + iy*nx + iz*nx*ny] - dt*k*(((DAT)1.0/dx)*(Vx[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]-Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )])
                                                                  + ((DAT)1.0/dy)*(Vy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]-Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)])
                                                                  + ((DAT)1.0/dz)*(Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]-Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]) ); }
}

int main(int argc, char *argv[]){
    int i, it;
    set_up_grid();
    set_up_parallelisation();
    if (me==0){ printf("Local sizes: Nx=%d, Ny=%d, Nz=%d, %d iterations. \n", nx,ny,nz,nt); }
    // Initial arrays
    zeros(x  ,nx  ,ny  ,nz  );
    zeros(y  ,nx  ,ny  ,nz  );
    zeros(z  ,nx  ,ny  ,nz  );
    zeros(P  ,nx  ,ny  ,nz  );
    zeros(Vx ,nx+1,ny  ,nz  );
    zeros(Vy ,nx  ,ny+1,nz  );
    zeros(Vz ,nx  ,ny  ,nz+1);
    // MPI sides
    init_sides(Vx ,nx+1,ny  ,nz  );
    init_sides(Vy ,nx  ,ny+1,nz  );
    init_sides(Vz ,nx  ,ny  ,nz+1);
    // Preprocessing
    Nix  = ((nx-2)*dims[0])+2;
    Niy  = ((ny-2)*dims[1])+2;
    Niz  = ((nz-2)*dims[2])+2;
    dx   = Lx/((DAT)Nix-(DAT)1.0);  // Global dx, dy
    dy   = Ly/((DAT)Niy-(DAT)1.0);
    dz   = Lz/((DAT)Niz-(DAT)1.0);
    DAT dt = min(min(dx,dy),dz)/sqrt(k/rho)/6.1;
    // Initial conditions
    int istep;
    init<<<grid,block>>>(x_d, y_d, z_d, P_d, coords_d, Lx, Ly, Lz, dx, dy, dz, nx, ny, nz); cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        if ((it%(int)5)==0){ MPI_Barrier(topo_comm); } // TEST
        if (it==10){ tic(); }
        // MPI overlap comm and compute
        for (istep=0; istep<2; istep++){
            compute_V<<<grid,block,0,streams[istep]>>>(Vx_d, Vy_d, Vz_d, P_d, dt, rho, dx, dy, dz, nx, ny, nz, istep);
            update_sides3(Vx,nx+1,ny,nz, Vy,nx,ny+1,nz, Vz,nx,ny,nz+1)
        }
        cudaDeviceSynchronize();
        compute_P<<<grid,block>>>(Vx_d, Vy_d, Vz_d, P_d, dt, k, dx, dy, dz, nx, ny, nz); cudaDeviceSynchronize();
    }//it
    tim("Performance", Nix*Niy*Niz*(nt-10)*8*PRECIS/(1e9)); // timer test
    free_all(x );
    free_all(y );
    free_all(z );
    free_all(P );
    free_all(Vx);
    free_all(Vy);
    free_all(Vz);
    // MPI
    free_sides(Vx);
    free_sides(Vy);
    free_sides(Vz);

    clean_cuda();
    MPI_Finalize();
    return 0;
}
