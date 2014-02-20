#ifndef STATS_CUH
#define STATS_CUH

#include <vector>
#include "cub/cub.cuh"
#include "cub_util.cuh"

template <class value_t, class index_t> __host__
void znormalize(value_t * const series, const index_t L) {
    
    value_t avg = 0, std=0;
    
    // calculate average
    for (int i = 0; i < L; ++i)
        avg += *(series+i);
    avg /= L;
    
    // calculate standard deviation
    for (int i = 0; i < L; ++i)
        std += (*(series+i))*(*(series+i));
    std = sqrt(std/L - avg*avg);
    
    // z-normalize the input
    for (int i = 0; i < L; ++i) {
        *(series+i) -= avg;
        *(series+i) /= std;
    }
}

template <class prefx_t, class value_t, class index_t>  __global__ 
void window(const prefx_t * const Input, 
            value_t * const Output, const index_t L, const index_t W) {

    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = tid; i < L-W+1; i += gridDim.x*blockDim.x)
        Output[i] = (Input[i+W]-Input[i]) / W;
}

template <class value_t, class index_t> __global__
void stddev(const value_t * const X, 
            const value_t * const X2,
            value_t * const Output, const index_t L, const index_t W) {

    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = tid; i < L-W+1; i += gridDim.x*blockDim.x) {
        value_t mu = X[i];
        mu = X2[i] - mu*mu;
        Output[i] = mu > 0 ? sqrt(mu) : 1;
    }
}

template <class prefx_t, class value_t, class index_t>  __global__ 
void plainCpy(const value_t * const Input, 
              prefx_t * const Output, const index_t L) {

    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = tid; i < L; i += gridDim.x*blockDim.x)
        Output[i] = Input[i];
}

template <class prefx_t, class value_t, class index_t>  __global__ 
void squareCpy(const value_t * const Input, 
               prefx_t * const Output, const index_t L) {

    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = tid; i < L; i += gridDim.x*blockDim.x) {
        const prefx_t value = Input[i];
        Output[i] = value*value;
    }
}

template <class index_t> __global__
void range(index_t * const Indices, const index_t L) {

    const int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (int i = thid; i < L; i += gridDim.x*blockDim.x)
        Indices[i] = i;
}

template <class prefx_t, class value_t, class index_t> __host__
void stats(const value_t * const Series,
           value_t * const AvgS, value_t * const StdS,
           const index_t M, const index_t N, cudaStream_t stream=0) {
             
    // create temporary storage for prefix sums
    prefx_t * Aprefix, * Sprefix;
    cudaMalloc(&Aprefix, sizeof(prefx_t)*(N+1));                          CUERR
    cudaMalloc(&Sprefix, sizeof(prefx_t)*(N+1));                          CUERR

    // convert data type with element-wise copy
    plainCpy  <<<GRIDDIM, BLOCKDIM, 0, stream>>> (Series, Aprefix, N);    CUERR
    squareCpy <<<GRIDDIM, BLOCKDIM, 0, stream>>> (Series, Sprefix, N);    CUERR

    // calculate prefix sums
    prefix_sum(Aprefix, Aprefix, N, stream);
    prefix_sum(Sprefix, Sprefix, N, stream);

    // calculate windowed difference (average)
    window<<<GRIDDIM, BLOCKDIM>>>(Aprefix, AvgS, N, M);                   CUERR
    
    // calculate windowed difference (standard deviation)
    window<<<GRIDDIM, BLOCKDIM>>>(Sprefix, StdS, N, M);                   CUERR
    stddev<<<GRIDDIM, BLOCKDIM>>>(AvgS, StdS, StdS, N, M);                CUERR

    // free memory
    cudaFree(Aprefix);                                                    CUERR
    cudaFree(Sprefix);                                                    CUERR
}

template <class prefx_t, class value_t, class index_t> __host__
void avg_std(const value_t * const Series,
           value_t * const AvgS, value_t * const StdS,
           const index_t M, const index_t N, cudaStream_t stream=0) {

    int epoch = 10000000;
    int width = epoch-M+1;
    std::vector<int> lowers = std::vector<int>();
    std::vector<int> uppers = std::vector<int>();
    
    for (int lower = 0; lower < N; lower += width) {
         int upper = min(lower+epoch, N);
         
         if (upper-lower > M+1) {
            lowers.push_back(lower);
            uppers.push_back(upper);
         } else {
            uppers[uppers.size()-1] = upper;
         }
    }
    
    for (unsigned int i = 0; i < lowers.size(); ++i) {
        int lower = lowers[i];
        int upper = uppers[i];
        stats<prefx_t>(Series+lower, AvgS+lower, StdS+lower,
                       M, upper-lower, stream);
    }
}

#endif
