#ifndef CUB_UTIL_CUH
#define CUB_UTIL_CUH

#include "cub/cub.cuh"
#include "cuda_def.cuh"

template <class value_t, class index_t, class F>
void reduce(value_t * input, value_t * output, 
            index_t L, F fn, cudaStream_t stream=0) {
    
    // determine temporary device storage requirements for reduction
    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    // memory for result    
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, 
                              input, output, L, fn, stream);              CUERR

    // allocate temporary storage for reduction
    cudaMalloc(&temp_storage, temp_storage_bytes);                        CUERR
    
    // run reduction
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, 
                              input, output, L, fn, stream);              CUERR
    
    cudaFree(temp_storage);                                               CUERR
}

template <class value_t, class index_t>
void prefix_sum(value_t * Input, value_t * Output,
                index_t L, cudaStream_t stream=0) {

    // init entry zero with zero
    value_t init = static_cast<value_t>(0);
    cudaMemcpyAsync(Output, &init, sizeof(value_t), 
                    cudaMemcpyHostToDevice, stream);                      CUERR

    // determine temporary device storage requirements for inclusive prefix scan
    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // run inclusive prefix scan dry to determine storage (sum)
    cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, 
                     Input, Output+1, cub::Sum(), L, stream);             CUERR

    // allocate temporary storage for inclusive prefix scan
    cudaMalloc(&temp_storage, temp_storage_bytes);                        CUERR

    // run inclusive prefix scan (sum)
    cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, 
                     Input, Output+1, cub::Sum(), L, stream);             CUERR

    cudaFree(temp_storage);                                               CUERR
}

template <class value_t, class index_t>
void inc_prefix_sum(value_t * Input, value_t * Output,
                    index_t L, cudaStream_t stream=0) {

    // determine temporary device storage requirements for inclusive prefix scan
    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // run inclusive prefix scan dry to determine storage (sum)
    cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, 
                     Input, Output, cub::Sum(), L, stream);               CUERR

    // allocate temporary storage for inclusive prefix scan
    cudaMalloc(&temp_storage, temp_storage_bytes);                        CUERR

    // run inclusive prefix scan (sum)
    cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, 
                     Input, Output, cub::Sum(), L, stream);               CUERR

    cudaFree(temp_storage);                                               CUERR
}


template <class value_t, class key_t, class index_t>
void pair_sort(key_t* Keys, value_t * Values, index_t L,
               cudaStream_t stream=0) {
        
    // create temporary double buffers
    key_t *kbuffer = NULL; value_t *vbuffer = NULL;
    
    cudaMalloc(&kbuffer, sizeof(key_t)*L);                                CUERR
    cudaMalloc(&vbuffer, sizeof(value_t)*L);                              CUERR
    cub::DoubleBuffer<key_t> d_keys(Keys, kbuffer);                       CUERR
    cub::DoubleBuffer<value_t> d_values(Values, vbuffer);                 CUERR
    
    // determine temporary device storage requirements for sorting operation
    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, 
         d_keys, d_values, L, 0, sizeof(key_t)*8, stream);                CUERR
    
    // allocate temporary storage for sorting operation
    cudaMalloc(&temp_storage, temp_storage_bytes);                        CUERR
    
    // run sorting operation
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, 
         d_keys, d_values, L, 0, sizeof(key_t)*8, stream);                CUERR
    
    // copy result
    cudaMemcpyAsync(Keys, d_keys.Current(), sizeof(value_t)*L, 
                    cudaMemcpyDeviceToDevice, stream);                    CUERR
    cudaMemcpyAsync(Values, d_values.Current(), sizeof(value_t)*L, 
                    cudaMemcpyDeviceToDevice, stream);                    CUERR

    // free memory
    cudaFree(kbuffer);                                                    CUERR
    cudaFree(vbuffer);                                                    CUERR
    cudaFree(temp_storage);                                               CUERR
}

template <class value_t, class index_t> __global__
void set_infty(value_t * Series, index_t L) {

    int thid = blockDim.x*blockIdx.x + threadIdx.x;

    if (thid < L)
        Series[thid] = INFINITY;
}

template <class value_t, class index_t> __global__
void thresh (value_t* Series, index_t* Mask, index_t L, value_t thr) {
    
    int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (thid < L)
        Mask[thid] = (Series[thid] <= thr);
}

template <class index_t> __global__
void merge (index_t* Mask, index_t* Prfx, index_t L) {
    
    int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (thid < L)
        Prfx[thid] *= Mask[thid];
}

template <class index_t> __global__
void collect(index_t* Indices, index_t* Mask, index_t* Prfx, index_t L) {

    int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (thid < L && Prfx[thid]) {
        Mask[Prfx[thid]-1] = Indices[thid];
    }

}

template <class value_t, class index_t> __host__
void threshold(value_t* Series, index_t* Indices, index_t L, index_t* newL,
              value_t bsf, cudaStream_t stream=0) {

    index_t *Mask = NULL, *Prfx = NULL;
    cudaMalloc(&Mask, sizeof(index_t)*L);                                 CUERR
    cudaMalloc(&Prfx, sizeof(index_t)*L);                                 CUERR
    thresh<<<SDIV(L, 1024), 1024, 0, stream>>>(Series, Mask, L, bsf);     CUERR
    inc_prefix_sum(Mask, Prfx, L, stream);                                CUERR
    cudaMemcpyAsync(newL, Prfx+L-1, sizeof(index_t), 
                    cudaMemcpyDeviceToHost, stream);                      CUERR
    merge<<<SDIV(L, 1024), 1024, 0, stream>>>(Mask, Prfx, L);             CUERR
    collect<<<SDIV(L, 1024), 1024, 0, stream>>>(Indices, Mask, Prfx, L);  CUERR
    cudaMemcpyAsync(Indices, Mask, sizeof(index_t)*L, 
                    cudaMemcpyDeviceToDevice, stream);                    CUERR

    cudaFree(Mask);                                                       CUERR
    cudaFree(Prfx);                                                       CUERR
}

#endif
