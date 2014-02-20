#ifndef CUDA_DEF_CUH
#define CUDA_DEF_CUH

#include <stdio.h>

#define CUERR { cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1);}}

#define SDIV(x,y) ((x+y-1)/y)

#define GRIDDIM (1024)
#define BLOCKDIM (1024)

#define STARTTIME cudaEvent_t Mstart, Mstop; \
                  float Mtime; \
                  cudaEventCreate(&Mstart); \
                  cudaEventCreate(&Mstop); \
                  cudaEventRecord(Mstart, 0);

#define STOPTIME cudaEventRecord(Mstop, 0); \
                 cudaEventSynchronize(Mstop); \
                 cudaEventElapsedTime(&Mtime, Mstart, Mstop); \
                 std::cout << "Mtime: " << Mtime << std::endl;


#endif
