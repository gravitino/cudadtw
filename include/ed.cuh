#ifndef ED_CUH
#define ED_CUH

#include <cufft.h>

#include "cub/cub.cuh"
#include "cub_util.cuh"
#include "stats.cuh"

#define MAXQUERYSIZE (8192)

__device__ __constant__ double Czquery[MAXQUERYSIZE];

template <class index_t> __global__
void embed_query(double * const Zquery, const index_t M, const index_t NN) {
    
    const int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = thid; i < NN; i += gridDim.x*blockDim.x)
        if (i < M)
            Zquery[i] = Czquery[i];
        else
            Zquery[i] = 0.0;
}

template <class index_t> __global__
void embed_subject(const double * const Subject, double * const Esubject,
                   const index_t N, const index_t NN) {
    
    const int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = thid; i < NN; i += gridDim.x*blockDim.x)
        if (i < N)
            Esubject[i] = Subject[i];
        else
            Esubject[i] = 0.0;
}

template <class index_t> __global__
void mult_conj(cufftDoubleComplex * const FftQ,
               const cufftDoubleComplex * const FftS, const index_t NNHPO) {

    const int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = thid; i < NNHPO; i += gridDim.x*blockDim.x) {
        
        // read entries from arrays
        const cufftDoubleComplex q = FftQ[i];
        const cufftDoubleComplex s = FftS[i];
                
        // calculate conjugate product
        cufftDoubleComplex r;
        r.x =  q.x*s.x + q.y*s.y;
        r.y = -q.y*s.x + q.x*s.y;
        
        // write result
        FftQ[i] = r;
    }
}

template <class index_t> __global__
void finalize(const double * const StdS, double * const Result, 
              const index_t M, const index_t N, const index_t NN) {
              
    const int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = thid; i < N-M+1; i += gridDim.x*blockDim.x) 
        Result[i] = -2*Result[i]/(StdS[i]*NN);
}

template <class index_t>
double calculate_ed(const double * const Subject, const double * const StdS, 
                    const index_t M, const index_t N, cudaStream_t stream=0) {

     // fixes some memory wasting of cufft
    const index_t NN =  SDIV(N, 10000)*10000; 

    // alloc space for ffts
    cufftDoubleComplex * FftQ = NULL, *FftS = NULL;
    cudaMalloc(&FftQ, sizeof(cufftDoubleComplex)*(NN/2+1));               CUERR
    cudaMalloc(&FftS, sizeof(cufftDoubleComplex)*(NN/2+1));               CUERR
    
    // embed query
    double * Equery = NULL;
    cudaMalloc(&Equery, sizeof(double)*NN);                               CUERR
    embed_query<<<GRIDDIM, BLOCKDIM, 0, stream>>>(Equery, M, NN);         CUERR
    
    // embed subject
    double * Esubject = NULL;
    cudaMalloc(&Esubject, sizeof(double)*NN);                             CUERR
    embed_subject<<<GRIDDIM, BLOCKDIM, 0, stream>>>
                 (Subject, Esubject, N, NN);                              CUERR
    
    // create plan for Hermitian forward transforms
    cufftHandle plan;
    cufftPlan1d(&plan, NN, CUFFT_D2Z, 1);
    cufftSetStream(plan, stream);
    
    // calculate rfft of embedded query
    cufftExecD2Z(plan, Equery, FftQ);                                     CUERR
    cudaThreadSynchronize();                                              CUERR

    // calculate rfft of subject
    cufftExecD2Z(plan, Esubject, FftS);                                   CUERR
    cudaThreadSynchronize();                                              CUERR
  
    // destroy plan  
    cufftDestroy(plan);
    
    // multiply in momentum space
    mult_conj<<<GRIDDIM, BLOCKDIM>>>(FftQ, FftS, NN/2+1);                 CUERR

    // create plan for Hermitian backward transforms
    cufftHandle inverse_plan;
    cufftPlan1d(&inverse_plan, NN, CUFFT_Z2D, 1);
    cufftSetStream(inverse_plan, stream);
   
    // calculate inverse rfft of product
    cufftExecZ2D(inverse_plan, FftQ, Equery);                             CUERR
    cudaThreadSynchronize();                                              CUERR
  
    // destroy plan  
    cufftDestroy(inverse_plan);

    // free temporary memory
    cudaFree(Esubject);                                                   CUERR
    cudaFree(FftS);                                                       CUERR
    cudaFree(FftQ);                                                       CUERR

    // calculate correl/std
    finalize<<<GRIDDIM, BLOCKDIM, 0, stream>>> (StdS, Equery, M, N, NN);  CUERR

    // indices for argsort
    int * Indices = NULL;
    cudaMalloc(&Indices, sizeof(int)*(N-M+1));                            CUERR
    range<<<GRIDDIM, BLOCKDIM, 0, stream>>> (Indices, N-M+1);             CUERR

    // argsort
    pair_sort(Equery, Indices, N-M+1, stream);

    double bsf = INFINITY; int bsf_index = -1;
    cudaMemcpyAsync(&bsf, Equery, sizeof(double), 
                    cudaMemcpyDeviceToHost, stream);                      CUERR
    cudaMemcpyAsync(&bsf_index, Indices, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);                      CUERR
    

    std::cout << "\n= result ===================================" << std::endl;
    std::cout << "distance: " << sqrt(2*M+bsf) << std::endl;
    std::cout << "location: " << bsf_index << std::endl;
    
    return bsf;
}

#endif
