// include file for MPI 3D geocomputing, 04.10.2017, Ludovic Raess
#include "mpi.h"

#ifdef USE_SINGLE_PRECISION
#define MPI_DAT  MPI_REAL
#else
#define MPI_DAT  MPI_DOUBLE_PRECISION
#endif

#define zeros_d(A,n1,n2)    DAT *A##_d; cudaMalloc(&A##_d,((n1)*(n2))*sizeof(DAT));
// #define NREQS               (2*2*NDIMS)
#define NREQS1              (2*2)
#define NREQS               (2*2*3)
#define neighbours(dim,nr)  __neighbours[nr + dim*2]
int dims[3]={DIMS_X,DIMS_Y,DIMS_Z};
int coords[3]={0,0,0};
int* coords_d=NULL;
int nprocs=-1, me_loc=-1, me=-1, gpu_id=-1;
int __neighbours[2*NDIMS]={-1}; // DEBUG neighbours(DIM.nr) macro in my case oposite to Sam's
int reqnr=0, tag=0;
int periods[NDIMS]={0};
int reorder=1;
MPI_Comm    topo_comm=MPI_COMM_NULL;
MPI_Request req[NREQS]={MPI_REQUEST_NULL};
// CommOverlap
cudaStream_t  streams[2];
#define CommOverlap() if ( istep==0 &&   ix>=BOUNDARY_WIDTH_X && ix<=(nx  )-1-BOUNDARY_WIDTH_X &&           \
                                         iy>=BOUNDARY_WIDTH_Y && iy<=(ny  )-1-BOUNDARY_WIDTH_Y &&           \
                                         iz>=BOUNDARY_WIDTH_Z && iz<=(nz  )-1-BOUNDARY_WIDTH_Z    ) return; \
                      if ( istep==1 && ( ix< BOUNDARY_WIDTH_X || ix> (nx  )-1-BOUNDARY_WIDTH_X ||           \
                                         iy< BOUNDARY_WIDTH_Y || iy> (ny  )-1-BOUNDARY_WIDTH_Y ||           \
                                         iz< BOUNDARY_WIDTH_Z || iz> (nz  )-1-BOUNDARY_WIDTH_Z  ) ) return;
                          
// Timer
#include "sys/time.h"
double timer_start = 0;
double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
double toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, double n){ double s=toc();if(me==0){ printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); } }

#define set_up_grid()  dim3 grid, block, grid_mpi0, block_mpi0, grid_mpi1, block_mpi1, grid_mpi2, block_mpi2; \
    block.x      = BLOCK_X; grid.x      = GRID_X; \
    block.y      = BLOCK_Y; grid.y      = GRID_Y; \
    block.z      = BLOCK_Z; grid.z      = GRID_Z; \
    block_mpi0.x = 1;       grid_mpi0.x = 1;      \
    block_mpi0.y = BLOCK_Y; grid_mpi0.y = GRID_Y; \
    block_mpi0.z = BLOCK_Z; grid_mpi0.z = GRID_Z; \
    block_mpi1.x = BLOCK_X; grid_mpi1.x = GRID_X; \
    block_mpi1.y = 1;       grid_mpi1.y = 1;      \
    block_mpi1.z = BLOCK_Z; grid_mpi1.z = GRID_Z; \
    block_mpi2.x = BLOCK_X; grid_mpi2.x = GRID_X; \
    block_mpi2.y = BLOCK_Y; grid_mpi2.y = GRID_Y; \
    block_mpi2.z = 1;       grid_mpi2.z = 1;

void  clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset(); }
    // CommOverlap
    cudaStreamDestroy(*streams);
    // cudaDeviceReset();
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
    gpu_id = me_loc; // GPU_ID
    cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    // CommOverlap
    int leastPriority=-1, greatestPriority=-1;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&streams[0], cudaStreamNonBlocking, greatestPriority);
    cudaStreamCreateWithPriority(&streams[1], cudaStreamNonBlocking, leastPriority);
    // MPI STUFF
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Dims_create(nprocs, NDIMS, dims);
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &topo_comm);
    MPI_Comm_rank(topo_comm, &me);
    MPI_Cart_coords(topo_comm, me, NDIMS, coords);
    cudaMalloc(&coords_d,3*sizeof(int)); cudaMemcpy(coords_d ,coords,3*sizeof(int),cudaMemcpyHostToDevice);
    for (int i=0; i<NDIMS; i++){ MPI_Cart_shift(topo_comm, i, 1, &(neighbours(i,0)), &(neighbours(i,1))); }
    if (me==0){ printf("MPI: nprocs=%d,dims(1)=%d,dims(2)=%d,dims(3)=%d \n", nprocs,dims[0],dims[1],dims[2]); }
}
#define set_up_parallelisation()  __set_up_parallelisation(argc, argv);

// MPI buffer init
__global__ void write_to_mpi_sendbuffer_00(DAT* A_send_00,DAT* A, const int nx_A, const int ny_A, const int nz_A, const int nx){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    ix =          ( (2+((nx_A)-nx))-1 );
    if (iz<(nz_A) && iy<(ny_A)){
        A_send_00[iy + iz*(ny_A)] = A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)]; }
}
__global__ void write_to_mpi_sendbuffer_01(DAT* A_send_01,DAT* A, const int nx_A, const int ny_A, const int nz_A, const int nx){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    ix = (nx_A)-1-( (2+((nx_A)-nx))-1 );
    if (iz<(nz_A) && iy<(ny_A)){
        A_send_01[iy + iz*(ny_A)] = A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)]; }
}
__global__ void write_to_mpi_sendbuffer_10(DAT* A_send_10,DAT* A, const int nx_A, const int ny_A, const int nz_A, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    iy =          ( (2+((ny_A)-ny))-1 );
    if (iz<(nz_A) && ix<(nx_A)){
        A_send_10[ix + iz*(nx_A)] = A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)]; }
}
__global__ void write_to_mpi_sendbuffer_11(DAT* A_send_11,DAT* A, const int nx_A, const int ny_A, const int nz_A, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    iy = (ny_A)-1-( (2+((ny_A)-ny))-1 );
    if (iz<(nz_A) && ix<(nx_A)){
        A_send_11[ix + iz*(nx_A)] = A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)]; }
}
__global__ void write_to_mpi_sendbuffer_20(DAT* A_send_20,DAT* A, const int nx_A, const int ny_A, const int nz_A, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz;
    iz =          ( (2+((nz_A)-nz))-1 );
    if (iy<(ny_A) && ix<(nx_A)){
        A_send_20[ix + iy*(nx_A)] = A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)]; }
}
__global__ void write_to_mpi_sendbuffer_21(DAT* A_send_21,DAT* A, const int nx_A, const int ny_A, const int nz_A, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz;
    iz = (nz_A)-1-( (2+((nz_A)-nz))-1 );
    if (iy<(ny_A) && ix<(nx_A)){
        A_send_21[ix + iy*(nx_A)] = A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)]; }
}
__global__ void read_from_mpi_recvbuffer_00(DAT* A,DAT* A_recv_00, const int nx_A, const int ny_A, const int nz_A){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    ix = 0;
    if (iz<(nz_A) && iy<(ny_A)){
        A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)] = A_recv_00[iy + iz*(ny_A)]; }
}
__global__ void read_from_mpi_recvbuffer_01(DAT* A,DAT* A_recv_01, const int nx_A, const int ny_A, const int nz_A){
    int ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    ix = (nx_A)-1;
    if (iz<(nz_A) && iy<(ny_A)){
        A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)] = A_recv_01[iy + iz*(ny_A)]; }
}
__global__ void read_from_mpi_recvbuffer_10(DAT* A,DAT* A_recv_10, const int nx_A, const int ny_A, const int nz_A){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    iy = 0;
    if (iz<(nz_A) && ix<(nx_A)){
        A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)] = A_recv_10[ix + iz*(nx_A)]; }
}
__global__ void read_from_mpi_recvbuffer_11(DAT* A,DAT* A_recv_11, const int nx_A, const int ny_A, const int nz_A){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy;
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    iy = (ny_A)-1;
    if (iz<(nz_A) && ix<(nx_A)){
        A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)] = A_recv_11[ix + iz*(nx_A)]; }
}
__global__ void read_from_mpi_recvbuffer_20(DAT* A,DAT* A_recv_20, const int nx_A, const int ny_A, const int nz_A){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz;
    iz = 0;
    if (iy<(ny_A) && ix<(nx_A)){
        A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)] = A_recv_20[ix + iy*(nx_A)]; }
}
__global__ void read_from_mpi_recvbuffer_21(DAT* A,DAT* A_recv_21, const int nx_A, const int ny_A, const int nz_A){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz;
    iz = (nz_A)-1;
    if (iy<(ny_A) && ix<(nx_A)){
        A[ix + iy*(nx_A) + iz*(nx_A)*(ny_A)] = A_recv_21[ix + iy*(nx_A)]; }
}

#define update_sides(A,nx_A,ny_A,nz_A) if (istep==0){ cudaDeviceSynchronize(); } else if (istep==1){ \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_send_01_d, A##_d, nx_A, ny_A, nz_A, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_00_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_01_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_send_00_d, A##_d, nx_A, ny_A, nz_A, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_01_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_00_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS1; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_d, A##_recv_00_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_d, A##_recv_01_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_send_11_d, A##_d, nx_A, ny_A, nz_A, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_10_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_11_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_send_10_d, A##_d, nx_A, ny_A, nz_A, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_11_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_10_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS1; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_d, A##_recv_10_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_d, A##_recv_11_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_send_21_d, A##_d, nx_A, ny_A, nz_A, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_20_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_21_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_send_20_d, A##_d, nx_A, ny_A, nz_A, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_21_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_20_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS1; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_d, A##_recv_20_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_d, A##_recv_21_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); }

#define update_sides3(A,nx_A,ny_A,nz_A, B,nx_B,ny_B,nz_B, C,nx_C,ny_C,nz_C)  if (istep==0){ cudaDeviceSynchronize(); } else if (istep==1){ \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_send_01_d, A##_d, nx_A, ny_A, nz_A, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(B##_send_01_d, B##_d, nx_B, ny_B, nz_B, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(C##_send_01_d, C##_d, nx_C, ny_C, nz_C, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_00_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Irecv(B##_recv_00_d, (ny_B)*(nz_B), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Irecv(C##_recv_00_d, (ny_C)*(nz_C), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_01_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Isend(B##_send_01_d, (ny_B)*(nz_B), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Isend(C##_send_01_d, (ny_C)*(nz_C), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_send_00_d, A##_d, nx_A, ny_A, nz_A, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(B##_send_00_d, B##_d, nx_B, ny_B, nz_B, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(C##_send_00_d, C##_d, nx_C, ny_C, nz_C, nx); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_01_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Irecv(B##_recv_01_d, (ny_B)*(nz_B), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,1) != MPI_PROC_NULL){  MPI_Irecv(C##_recv_01_d, (ny_C)*(nz_C), MPI_DAT, neighbours(0,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_00_d, (ny_A)*(nz_A), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Isend(B##_send_00_d, (ny_B)*(nz_B), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(0,0) != MPI_PROC_NULL){  MPI_Isend(C##_send_00_d, (ny_C)*(nz_C), MPI_DAT, neighbours(0,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_d, A##_recv_00_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(B##_d, B##_recv_00_d, nx_B, ny_B, nz_B); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_00<<<grid_mpi0,block_mpi0,0,streams[0]>>>(C##_d, C##_recv_00_d, nx_C, ny_C, nz_C); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(A##_d, A##_recv_01_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(B##_d, B##_recv_01_d, nx_B, ny_B, nz_B); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(0,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_01<<<grid_mpi0,block_mpi0,0,streams[0]>>>(C##_d, C##_recv_01_d, nx_C, ny_C, nz_C); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_send_11_d, A##_d, nx_A, ny_A, nz_A, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(B##_send_11_d, B##_d, nx_B, ny_B, nz_B, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(C##_send_11_d, C##_d, nx_C, ny_C, nz_C, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_10_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Irecv(B##_recv_10_d, (nx_B)*(nz_B), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Irecv(C##_recv_10_d, (nx_C)*(nz_C), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_11_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Isend(B##_send_11_d, (nx_B)*(nz_B), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Isend(C##_send_11_d, (nx_C)*(nz_C), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_send_10_d, A##_d, nx_A, ny_A, nz_A, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(B##_send_10_d, B##_d, nx_B, ny_B, nz_B, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(C##_send_10_d, C##_d, nx_C, ny_C, nz_C, ny); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_11_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Irecv(B##_recv_11_d, (nx_B)*(nz_B), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,1) != MPI_PROC_NULL){  MPI_Irecv(C##_recv_11_d, (nx_C)*(nz_C), MPI_DAT, neighbours(1,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_10_d, (nx_A)*(nz_A), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Isend(B##_send_10_d, (nx_B)*(nz_B), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(1,0) != MPI_PROC_NULL){  MPI_Isend(C##_send_10_d, (nx_C)*(nz_C), MPI_DAT, neighbours(1,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_d, A##_recv_10_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(B##_d, B##_recv_10_d, nx_B, ny_B, nz_B); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_10<<<grid_mpi1,block_mpi1,0,streams[0]>>>(C##_d, C##_recv_10_d, nx_C, ny_C, nz_C); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(A##_d, A##_recv_11_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(B##_d, B##_recv_11_d, nx_B, ny_B, nz_B); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(1,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_11<<<grid_mpi1,block_mpi1,0,streams[0]>>>(C##_d, C##_recv_11_d, nx_C, ny_C, nz_C); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_send_21_d, A##_d, nx_A, ny_A, nz_A, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(B##_send_21_d, B##_d, nx_B, ny_B, nz_B, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(C##_send_21_d, C##_d, nx_C, ny_C, nz_C, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_20_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Irecv(B##_recv_20_d, (nx_B)*(ny_B), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Irecv(C##_recv_20_d, (nx_C)*(ny_C), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Isend(A##_send_21_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Isend(B##_send_21_d, (nx_B)*(ny_B), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Isend(C##_send_21_d, (nx_C)*(ny_C), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_send_20_d, A##_d, nx_A, ny_A, nz_A, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(B##_send_20_d, B##_d, nx_B, ny_B, nz_B, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)    write_to_mpi_sendbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(C##_send_20_d, C##_d, nx_C, ny_C, nz_C, nz); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Irecv(A##_recv_21_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Irecv(B##_recv_21_d, (nx_B)*(ny_B), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,1) != MPI_PROC_NULL){  MPI_Irecv(C##_recv_21_d, (nx_C)*(ny_C), MPI_DAT, neighbours(2,1), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Isend(A##_send_20_d, (nx_A)*(ny_A), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Isend(B##_send_20_d, (nx_B)*(ny_B), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       if (neighbours(2,0) != MPI_PROC_NULL){  MPI_Isend(C##_send_20_d, (nx_C)*(ny_C), MPI_DAT, neighbours(2,0), tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
                                       MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS; j++){ req[j]=MPI_REQUEST_NULL; }; \
                                       cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_d, A##_recv_20_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(B##_d, B##_recv_20_d, nx_B, ny_B, nz_B); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,0) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_20<<<grid_mpi2,block_mpi2,0,streams[0]>>>(C##_d, C##_recv_20_d, nx_C, ny_C, nz_C); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(A##_d, A##_recv_21_d, nx_A, ny_A, nz_A); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(B##_d, B##_recv_21_d, nx_B, ny_B, nz_B); cudaStreamSynchronize(streams[0]); \
                                       if (neighbours(2,1) != MPI_PROC_NULL)   read_from_mpi_recvbuffer_21<<<grid_mpi2,block_mpi2,0,streams[0]>>>(C##_d, C##_recv_21_d, nx_C, ny_C, nz_C); cudaStreamSynchronize(streams[0]); }

#define init_sides(A,nx_A,ny_A,nz_A)   zeros_d(A##_send_00 ,ny_A,nz_A); zeros_d(A##_send_01 ,ny_A,nz_A); zeros_d(A##_send_10 ,nx_A,nz_A); zeros_d(A##_send_11 ,nx_A,nz_A); zeros_d(A##_send_20 ,nx_A,ny_A); zeros_d(A##_send_21 ,nx_A,ny_A); zeros_d(A##_recv_00 ,ny_A,nz_A); zeros_d(A##_recv_01 ,ny_A,nz_A); zeros_d(A##_recv_10 ,nx_A,nz_A); zeros_d(A##_recv_11 ,nx_A,nz_A); zeros_d(A##_recv_20 ,nx_A,ny_A); zeros_d(A##_recv_21 ,nx_A,ny_A);
#define free_sides(A)                  cudaFree(A##_send_00_d); cudaFree(A##_send_01_d); cudaFree(A##_send_10_d); cudaFree(A##_send_11_d); cudaFree(A##_send_20_d); cudaFree(A##_send_21_d); cudaFree(A##_recv_00_d); cudaFree(A##_recv_01_d); cudaFree(A##_recv_10_d); cudaFree(A##_recv_11_d); cudaFree(A##_recv_20_d); cudaFree(A##_recv_21_d);
