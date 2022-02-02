// include file for MPI 2D geocomputing, 04.10.2017, Ludovic Raess
#include "mpi.h"

#ifdef USE_SINGLE_PRECISION
#define MPI_DAT  MPI_REAL
#else
#define MPI_DAT  MPI_DOUBLE_PRECISION
#endif

#define zeros_d(A,nxy)      DAT *A##_d; cudaMalloc(&A##_d,(nxy)*sizeof(DAT));
#define NREQS               (2*2*NDIMS)
#define neighbours(dim,nr)  __neighbours[nr + dim*2]
int dims[3]={DIMS_X,DIMS_Y,1};
int coords[3]={0,0,0};
int* coords_d=NULL;
int nprocs=-1, me=-1, me_loc=-1, gpu_id=-1;
int __neighbours[2*NDIMS]={-1}; // DEBUG neighbours(DIM.nr) macro in my case oposite to Sam's
int reqnr=0, tag=0;
int periods[NDIMS]={0};
int reorder=1;
MPI_Comm    topo_comm=MPI_COMM_NULL;
MPI_Request req[NREQS]={MPI_REQUEST_NULL};

// Timer
#include "sys/time.h"
double timer_start = 0;
double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
double toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, double n){ double s=toc();if(me==0){ printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); } }

#define set_up_grid()  dim3 grid, block, grid_mpi0, block_mpi0, grid_mpi1, block_mpi1; \
    block.x      = BLOCK_X; grid.x      = GRID_X; \
    block.y      = BLOCK_Y; grid.y      = GRID_Y; \
    block_mpi0.x = 1;       grid_mpi0.x = 1;      \
    block_mpi0.y = BLOCK_Y; grid_mpi0.y = GRID_Y; \
    block_mpi1.x = BLOCK_X; grid_mpi1.x = GRID_X; \
    block_mpi1.y = 1;       grid_mpi1.y = 1;

void  clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset(); }
    cudaDeviceReset();
}

void __set_up_parallelisation(int argc, char *argv[]){
    // GPU STUFF
    cudaSetDeviceFlags(cudaDeviceMapHost); // DEBUG: needs to be set before context creation !
    const char* me_str     = getenv("OMPI_COMM_WORLD_RANK");
    const char* me_loc_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    // const char* me_str     = getenv("MV2_COMM_WORLD_RANK");
    // const char* me_loc_str = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    me     = atoi(me_str);
    me_loc = atoi(me_loc_str);
    gpu_id = me_loc;
    cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    // MPI STUFF
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Dims_create(nprocs, NDIMS, dims);
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &topo_comm);
    MPI_Comm_rank(topo_comm, &me);
    MPI_Cart_coords(topo_comm, me, NDIMS, coords);
    cudaMalloc(&coords_d,3*sizeof(int)); cudaMemcpy(coords_d ,coords,3*sizeof(int),cudaMemcpyHostToDevice);
    for (int i=0; i<NDIMS; i++){ MPI_Cart_shift(topo_comm, i, 1, &(neighbours(i,0)), &(neighbours(i,1))); }
    if (me==0){ printf("nprocs=%d,dims(1)=%d,dims(2)=%d \n", nprocs,dims[0],dims[1]); }
}
#define set_up_parallelisation()  __set_up_parallelisation(argc, argv);

// MPI buffer init
__global__ void write_to_mpi_sendbuffer_00(DAT* A_send_00,DAT* A, const int nx_A, const int ny_A, const int nx){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    ix =          ( (2+((nx_A)-nx))-1 );
    if (iy<(ny_A)){
        A_send_00[iy] = A[ix + iy*(nx_A)]; }
}
__global__ void write_to_mpi_sendbuffer_01(DAT* A_send_01,DAT* A, const int nx_A, const int ny_A, const int nx){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    ix = (nx_A)-1-( (2+((nx_A)-nx))-1 );
    if (iy<(ny_A)){
        A_send_01[iy] = A[ix + iy*(nx_A)]; }
}
__global__ void write_to_mpi_sendbuffer_10(DAT* A_send_10,DAT* A, const int nx_A, const int ny_A, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    iy =          ( (2+((ny_A)-ny))-1 );
    if (ix<(nx_A)){
        A_send_10[ix] = A[ix + iy*(nx_A)]; }
}
__global__ void write_to_mpi_sendbuffer_11(DAT* A_send_11,DAT* A, const int nx_A, const int ny_A, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    iy = (ny_A)-1-( (2+((ny_A)-ny))-1 );
    if (ix<(nx_A)){
        A_send_11[ix] = A[ix + iy*(nx_A)]; }
}
__global__ void read_from_mpi_recvbuffer_00(DAT* A,DAT* A_recv_00, const int nx_A, const int ny_A){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    ix = 0;
    if (iy<(ny_A)){
        A[ix + iy*(nx_A)] = A_recv_00[iy]; }
}
__global__ void read_from_mpi_recvbuffer_01(DAT* A,DAT* A_recv_01, const int nx_A, const int ny_A){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    ix = (nx_A)-1;
    if (iy<(ny_A)){
        A[ix + iy*(nx_A)] = A_recv_01[iy]; }
}
__global__ void read_from_mpi_recvbuffer_10(DAT* A,DAT* A_recv_10, const int nx_A, const int ny_A){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    iy = 0;
    if (ix<(nx_A)){
        A[ix + iy*(nx_A)] = A_recv_10[ix]; }
}
__global__ void read_from_mpi_recvbuffer_11(DAT* A,DAT* A_recv_11, const int nx_A, const int ny_A){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    iy = (ny_A)-1;
    if (ix<(nx_A)){
        A[ix + iy*(nx_A)] = A_recv_11[ix]; }
}

#define update_sides(A,nx_A,ny_A) cudaDeviceSynchronize(); \
                                  if (neighbours(0,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_01<<<grid_mpi0,block_mpi0>>>(A##_send_01_d, A##_d, nx_A, ny_A, nx); cudaDeviceSynchronize(); \
                                  if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_00_d, (ny_A), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_01_d, (ny_A), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  cudaDeviceSynchronize(); \
                                  if (neighbours(0,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_00<<<grid_mpi0,block_mpi0>>>(A##_send_00_d, A##_d, nx_A, ny_A, nx); cudaDeviceSynchronize(); \
                                  if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_01_d, (ny_A), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_00_d, (ny_A), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                  cudaDeviceSynchronize(); \
                                  if (neighbours(0,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_00<<<grid_mpi0,block_mpi0>>>(A##_d, A##_recv_00_d, nx_A, ny_A); cudaDeviceSynchronize(); \
                                  if (neighbours(0,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_01<<<grid_mpi0,block_mpi0>>>(A##_d, A##_recv_01_d, nx_A, ny_A); cudaDeviceSynchronize(); \
                                  if (neighbours(1,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_11<<<grid_mpi1,block_mpi1>>>(A##_send_11_d, A##_d, nx_A, ny_A, ny); cudaDeviceSynchronize(); \
                                  if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_10_d, (nx_A), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_11_d, (nx_A), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  cudaDeviceSynchronize(); \
                                  if (neighbours(1,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_10<<<grid_mpi1,block_mpi1>>>(A##_send_10_d, A##_d, nx_A, ny_A, ny); cudaDeviceSynchronize(); \
                                  if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_11_d, (nx_A), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_10_d, (nx_A), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                  MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                  cudaDeviceSynchronize(); \
                                  if (neighbours(1,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_10<<<grid_mpi1,block_mpi1>>>(A##_d, A##_recv_10_d, nx_A, ny_A); cudaDeviceSynchronize(); \
                                  if (neighbours(1,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_11<<<grid_mpi1,block_mpi1>>>(A##_d, A##_recv_11_d, nx_A, ny_A); cudaDeviceSynchronize();

#define init_sides(A,nx_A,ny_A)   zeros_d(A##_send_00 ,ny_A); zeros_d(A##_send_01 ,ny_A); zeros_d(A##_send_10 ,nx_A); zeros_d(A##_send_11 ,nx_A); zeros_d(A##_recv_00 ,ny_A); zeros_d(A##_recv_01 ,ny_A); zeros_d(A##_recv_10 ,nx_A); zeros_d(A##_recv_11 ,nx_A);
#define free_sides(A)             cudaFree(A##_send_00_d); cudaFree(A##_send_01_d); cudaFree(A##_send_10_d); cudaFree(A##_send_11_d); cudaFree(A##_recv_00_d); cudaFree(A##_recv_01_d); cudaFree(A##_recv_10_d); cudaFree(A##_recv_11_d);
