#ifndef CDTW_CUH
#define CDTW_CUH

#include <omp.h>                // omp pragmas
#include "cuda_def.cuh"         // CUERR makro
#include "stats.cuh"            // statistics of time series

#define MAXQUERYSIZE (4096)
__device__ __constant__ float Czlower[MAXQUERYSIZE];
__device__ __constant__ float Czupper[MAXQUERYSIZE];
__device__ __constant__ float Czquery[MAXQUERYSIZE];

// bind texture to subject to enhance performance on access
texture<float, 1, cudaReadModeElementType> Tsubject;

// insert code for lower bounds here
#include "bounds.cuh"

///////////////////////////////////////////////////////////////////////////////
// CPU constrained DTWs
///////////////////////////////////////////////////////////////////////////////

// linear memory constrained block dtw (sequential cpu)
template <bool euclidean, class value_t, class index_t> __host__
void sequential_cdtw(value_t* zquery, value_t* subject, value_t* avgS, 
                     value_t* stdS, value_t* cdtw, index_t M, index_t* indices, 
                     index_t indices_l, index_t W){

    int lane = M+1;
    value_t * penalty = new value_t[2*lane];

    for (int seid = 0; seid < indices_l; ++seid) {
    
        // initialize penalty matrix
        penalty[0] = 0;
        for (int m = 1; m < lane; ++m)
            penalty[m] = INFINITY;
            
        // relax nodes
        int src_row = 1, trg_row = 0;
    
        // get statistics
        int indx = indices[seid];
        value_t avg = avgS[indx];
        value_t std = stdS[indx];
        
        for (int i = 1; i < lane; ++i) {
        
            // get source and target row via cyclic indexing
            trg_row = src_row;
            src_row = (trg_row == 0);
        
            // get borders of band
            int lower = max(0, i-W-1), upper = min(lane, i+W+1);
        
            // set left halo pixel
            penalty[trg_row*lane+lower] = INFINITY;
    
            // relax each cell in the target row
            for (int j = lower+1; j < upper; ++j) {
            
                // accumulate result over all dimensions
                value_t weight = 0;            
            
                value_t value = zquery[i-1]-(subject[indx+j-1]-avg)/std;
                weight += euclidean ? value*value : value < 0 ? -value : value;

                // relax all three incoming edges
                weight += min(penalty[src_row*lane+j-1],
                          min(penalty[src_row*lane+j],
                              penalty[trg_row*lane+j-1]));
                         
                // write down result
                penalty[trg_row*lane+j] = weight;
            }
        
            // set right halo pixel
            if (upper < lane)
                penalty[trg_row*lane+upper] = INFINITY;
        }
        
        // write down result
        cdtw[seid] = penalty[trg_row*lane+M];
    }
    
    // free memory
    delete [] penalty;
}

// linear memory constrained block dtw (openmp cpu)
template <bool euclidean, class value_t, class index_t> __host__
void openmp_cdtw(value_t* zquery, value_t* subject, value_t* avgS, 
                     value_t* stdS, value_t* cdtw, index_t M, index_t* indices, 
                     index_t indices_l, index_t W){

    int lane = M+1;

    # pragma omp parallel for
    for (int seid = 0; seid < indices_l; ++seid) {
    
        value_t * penalty = new value_t[2*lane];
        
        // initialize penalty matrix
        penalty[0] = 0;
        for (int m = 1; m < lane; ++m)
            penalty[m] = INFINITY;
            
        // relax nodes
        int src_row = 1, trg_row = 0;
    
        // get statistics
        int indx = indices[seid];
        value_t avg = avgS[indx];
        value_t std = stdS[indx];
        
        for (int i = 1; i < lane; ++i) {
        
            // get source and target row via cyclic indexing
            trg_row = src_row;
            src_row = (trg_row == 0);
        
            // get borders of band
            int lower = max(0, i-W-1), upper = min(lane, i+W+1);
        
            // set left halo pixel
            penalty[trg_row*lane+lower] = INFINITY;
    
            // relax each cell in the target row
            for (int j = lower+1; j < upper; ++j) {
            
                // accumulate result over all dimensions
                value_t weight = 0;            
            
                value_t value = zquery[i-1]-(subject[indx+j-1]-avg)/std;
                weight += euclidean ? value*value : value < 0 ? -value : value;

                // relax all three incoming edges
                weight += min(penalty[src_row*lane+j-1],
                          min(penalty[src_row*lane+j],
                              penalty[trg_row*lane+j-1]));
                         
                // write down result
                penalty[trg_row*lane+j] = weight;
            }
        
            // set right halo pixel
            if (upper < lane)
                penalty[trg_row*lane+upper] = INFINITY;
        }
        
        // write down result
        cdtw[seid] = penalty[trg_row*lane+M];
        
        // free memory
        delete [] penalty;
    }
}

///////////////////////////////////////////////////////////////////////////////
// GPU constrained DTWs
///////////////////////////////////////////////////////////////////////////////

// linear memory constrained block dtw (one dtw per block)
template <bool euclidean, class value_t, class index_t> __global__
void sparse_block_cdtw(value_t* Subject, value_t* AvgS, value_t* StdS, 
                       value_t* Cdtw, index_t M, index_t* Indices, 
                       index_t indices_l, index_t W){

    int thid = threadIdx.x;
    int seid = blockIdx.x; 

    // retrieve actual index and lane length
    int indx = Indices[seid];
    int lane = W+2;
    int cntr = (W+1)/2;

    extern __shared__ value_t cache[];

    value_t * sh_Penalty = cache;
    value_t * sh_Query = cache+3*lane;
    
    if (seid < indices_l) {
        
        // retrieve statistics of subject subsequence
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];
        
        // initialize penalty matrix
        for (int m = thid; m < 3*lane; m += blockDim.x)
            sh_Penalty[m] = INFINITY;
        sh_Penalty[cntr] = 0;
        
        // initialize shared memory for query
        for (int m =  thid; m < M; m += blockDim.x)
            sh_Query[m] = Czquery[m];

        __syncthreads();

        // initialize diagonal pattern
        int p = W & 1;
        int q = (p == 0);
        
        // k % 2 and k / 2
        int km2 = 1, kd2 = 0;
        
        // row indices
        int trg_row = 1, pr1_row = 0, pr2_row = 2;
        
        // relax diagonal pattern
        for (int k = 2; k < 2*M+1; ++k) {

            // base index of row and associated shift
            kd2 += km2;
            km2 = (km2 == 0);
            int b_i = cntr+kd2+q*km2, b_j = -cntr+kd2+p*km2, shift = (p^km2);
            
            // cyclic indexing of rows
            pr1_row = trg_row;
            pr2_row = trg_row == 0 ? 2 : trg_row - 1;
            trg_row = trg_row == 2 ? 0 : trg_row + 1;
            
            for (int l = thid; l < lane-1; l += blockDim.x) {
                
                // potential index of node (if valid)
                int i = b_i-l;
                int j = b_j+l;
                
                // check if in valid relaxation area
                bool inside = 0 < i && i < M+1 && 0 < j && j < M+1 &&
                              shift <= l && l < lane-1;
                
                value_t value = INFINITY;
                
                if (inside) {
                    value = (tex1Dfetch(Tsubject, indx+j-1)-avg)/std 
                          - sh_Query[i-1];
                    
                    // Euclidean or Manhattan distance?
                    value = euclidean ? value*value :
                            value < 0 ? -value : value;

                    // get mininum incoming edge and relax
                    value += min(sh_Penalty[pr2_row*lane+l],
                             min(sh_Penalty[pr1_row*lane+l-shift],
                                 sh_Penalty[pr1_row*lane+l-shift+1]));
                }
                
                // write value to penalty matrix
                sh_Penalty[trg_row*lane+l] = value;
            }
            // sync threads after each row
             __syncthreads();
        }
        
        // write result down
        Cdtw[seid] = sh_Penalty[((2*M) % 3)*lane+cntr];
    }
}


// linear memory constrained block dtw (many dtws per block)
template <bool euclidean, class value_t, class index_t> __global__
void dense_block_cdtw(value_t* Subject, value_t* AvgS, value_t* StdS, 
                      value_t* Cdtw, index_t M, index_t* Indices, 
                      index_t indices_l, index_t W){

    // lane length and center node
    const int lane = W+2;
    const int cntr = (W+1)/2;
    
    // number of dtws per block, slot id of dtw and global id
    const int size = blockDim.x/lane;
    const int slid = threadIdx.x/lane;
    const int nidx = threadIdx.x - slid*lane;
    const int seid = size*blockIdx.x + slid;

    extern __shared__ value_t cache[];

    // set memory location in shared memory
    value_t * sh_Query = cache;
    value_t * sh_Penalty = sh_Query+M+slid*(3*lane);

    // initialize shared memory for query
    for (int m =  threadIdx.x; m < M; m += blockDim.x)
        sh_Query[m] = Czquery[m];

    // initialize penalty matrix
    for (int m = threadIdx.x; m < 3*lane*size; m += blockDim.x)
        cache[M+m] = INFINITY;
    __syncthreads();
   
    // set upper left cell to zero
    if (nidx == 0)
        sh_Penalty[cntr] = 0;
    __syncthreads();

    if (seid < indices_l) {
        
        // retrieve statistics of subject subsequence
        int indx = Indices[seid];
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];

        // initialize diagonal pattern
        int p = W & 1;
        int q = (p == 0);
        
        // k % 2 and k / 2
        int km2 = 1, kd2 = 0;
        
        // row indices
        int trg_row = 1, pr1_row = 0, pr2_row = 2;
        
        // relax diagonal pattern
        for (int k = 2; k < 2*M+1; ++k) {

            // base index of row and associated shift
            kd2 += km2;
            km2 = (km2 == 0);
            int b_i = cntr+kd2+q*km2, b_j = -cntr+kd2+p*km2, shift = (p^km2);
            
            // cyclic indexing of rows
            pr1_row = trg_row;
            pr2_row = trg_row == 0 ? 2 : trg_row - 1;
            trg_row = trg_row == 2 ? 0 : trg_row + 1;
            
            for (int l = nidx; l < lane-1; l += blockDim.x) {
                
                // potential index of node (if valid)
                int i = b_i-l;
                int j = b_j+l;
                
                // check if in valid relaxation area
                bool inside = 0 < i && i < M+1 && 0 < j && j < M+1 &&
                              shift <= l && l < lane-1;
                
                value_t value = INFINITY;
                
                if (inside) {
                    value = (tex1Dfetch(Tsubject, indx+j-1)-avg)/std 
                          - sh_Query[i-1];
                    
                    // Euclidean or Manhattan distance?
                    value = euclidean ? value*value :
                            value < 0 ? -value : value;

                    // get mininum incoming edge and relax
                    value += min(sh_Penalty[pr2_row*lane+l],
                             min(sh_Penalty[pr1_row*lane+l-shift],
                                 sh_Penalty[pr1_row*lane+l-shift+1]));
                }
                
                // write value to penalty matrix
                sh_Penalty[trg_row*lane+l] = value;
            }
            // sync threads after each row
             __syncthreads();
        }
        
        // write result down
        Cdtw[seid] = sh_Penalty[((2*M) % 3)*lane+cntr];
    }
}

// linear memory constrained dtw (one dtw per thread)
template <bool euclidean, class value_t, class index_t> __global__
void thread_cdtw(value_t* Subject, value_t* AvgS, value_t* StdS, value_t* Cdtw,
                 index_t M, index_t* Indices, index_t indices_l, index_t W) {

    int thid = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (thid < indices_l) {
        
        
        // retrieve actual index and lane length
        int indx = Indices[thid];
        int lane = M+1;
        
        // retrieve statistics of subject subsequence
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];
        
        // linear memory algorithm M(n) = 2*(n+1) in O(n)
        value_t penalty[2*(MAXQUERYSIZE+1)];
       
        // initialize first row of penalty matrix
        penalty[0] = 0;
        for (int j = 1; j < lane; ++j)
            penalty[j] = INFINITY;

        // relax row-wise in topological sorted order
        for (int i = 1, src_row = 1, trg_row = 0; i < lane; ++i) {
            
            // calculate indices of source and target row
            trg_row = src_row;
            src_row = (trg_row == 0);
            int src_offset = src_row*lane;
            int trg_offset = trg_row*lane;
            value_t zquery_value = Czquery[i-1];
            
            // initialize target row
            for (int j = 0; j < lane; ++j)
                penalty[trg_offset+j] = INFINITY;
            
            // calculate Sakoe-Chiba band and relax nodes
            int lower = max(i-W, 1);
            int upper = min(i+W+1, lane);
           
            for (int j = lower; j < upper; ++j) {
                 
                 value_t value = zquery_value -
                                 (tex1Dfetch(Tsubject, indx+j-1)-avg)/std;
                 
                 // Euclidean or Manhattan distance?
                 value = euclidean ? value*value :
                         value < 0 ? -value : value;
                 
                 // get nodes from the three incoming edges
                 value += min(penalty[src_offset+j-1], 
                          min(penalty[src_offset+j],
                              penalty[trg_offset+j-1]));
                 
                 // backrelax the three incoming edges
                 penalty[trg_offset+j] = value;
            }
        }
        
        // write down result
        Cdtw[thid] = penalty[(M & 1)*lane+M]; 
    }
}

///////////////////////////////////////////////////////////////////////////////
// Wrappers
///////////////////////////////////////////////////////////////////////////////

template <bool euclidean, class value_t, class index_t> __host__
void cpu_cdtw(value_t* zquery, value_t* subject, value_t* avgS, 
              value_t* stdS, value_t* cdtw, index_t M, index_t* indices, 
              index_t indices_l, index_t W, bool openmp = true) {

    if (openmp)
        openmp_cdtw<euclidean>
        (zquery, subject, avgS, stdS, cdtw, M, indices, indices_l, W);
    else
        sequential_cdtw<euclidean>
        (zquery, subject, avgS, stdS, cdtw, M, indices, indices_l, W);
}

template <bool euclidean, class value_t, class index_t> __host__
void gpu_cdtw(value_t* Subject, value_t* AvgS, value_t* StdS, 
              value_t* Cdtw, index_t M, index_t* Indices, index_t indices_l, 
              index_t W, bool blockdtw = true, cudaStream_t stream=0){

    if (blockdtw) {
        int lane = W+2;
        int size = 256/lane;

        if ((3*lane+M)*sizeof(float) > 48*1024 or lane > 1024) {
            std::cout << "ERROR: Not enough shared memory or threads present: "
                      << "please decrease query or window size. "
                      << "Alternatively use threaded DTW (blockdtw=false)."
                      << std::endl;
            return;
        }

        if (size == 0) {
            sparse_block_cdtw <euclidean> 
            <<<indices_l, lane, (3*lane+M)*sizeof(float), stream>>>
            (Subject, AvgS, StdS, Cdtw, M, Indices, indices_l, W);        CUERR
        } else {
            int blockdim = size*lane;
            int griddim = SDIV(indices_l, size);
            
            dense_block_cdtw <euclidean> 
            <<<griddim, blockdim, (M+size*3*lane)*sizeof(float), stream>>>
            (Subject, AvgS, StdS, Cdtw, M, Indices, indices_l, W);        CUERR
        }

    } else {
        int blockdim = 64;
        int griddim = SDIV(indices_l, blockdim);
        
        thread_cdtw<euclidean> <<<griddim, blockdim, 0, stream>>>
        (Subject, AvgS, StdS, Cdtw, M, Indices, indices_l, W);            CUERR
    }
}

///////////////////////////////////////////////////////////////////////////////
// GPU lower bounded condstrained DTW
///////////////////////////////////////////////////////////////////////////////

template <bool euclidean, class value_t, class index_t> __host__
void report_result(value_t * Cdtw, index_t * Indices, index_t upper, 
                   cudaStream_t stream = 0) {

    pair_sort(Cdtw, Indices, upper, stream);
    value_t bsf = INFINITY;
    index_t bsf_index = -1;
    
    cudaMemcpyAsync(&bsf, Cdtw, sizeof(value_t), 
                    cudaMemcpyDeviceToHost, stream);                      CUERR
    cudaMemcpyAsync(&bsf_index, Indices, sizeof(index_t),
                    cudaMemcpyDeviceToHost, stream);                      CUERR
    
    std:: cout << "Location: " << bsf_index << std::endl;
    std:: cout << "Distance: " << (euclidean ? sqrt(bsf) : bsf) << std::endl;
}

template <bool euclidean, class value_t, class index_t> __host__
void prune_cdtw(value_t* Subject, value_t* AvgS, value_t* StdS, index_t M, 
                index_t N, index_t W, cudaStream_t stream=0) {


    
    value_t *Lb_kim = NULL, *Lb_keogh = NULL, *Cdtw = NULL, *Best_cdtw = NULL,
            max_kim = INFINITY, best_cdtw = INFINITY, tmp = INFINITY;
    
    index_t *Indices = NULL;
    
    cudaMalloc(&Best_cdtw, sizeof(value_t));                              CUERR
    cudaMalloc(&Lb_kim, sizeof(value_t)*(N-M+1));                         CUERR
    cudaMalloc(&Lb_keogh, sizeof(value_t)*(N-M+1));                       CUERR
    cudaMalloc(&Cdtw, sizeof(value_t)*(N-M+1));                           CUERR
    cudaMalloc(&Indices, sizeof(value_t)*(N-M+1));                        CUERR
    
    // init Cdtw with infinity
    set_infty <<<SDIV(N-M+1, 1024), 1024, 0, stream>>> (Cdtw, N-M+1);     CUERR
    
    // calculate LB_Kim for all entries and sort indices
    lb_kim<euclidean>(Subject, AvgS, StdS, Lb_kim, M, N, stream);         CUERR
    range<<<1024, 1024, 0, stream>>>(Indices, N-M+1);                     CUERR
    pair_sort(Lb_kim, Indices, N-M+1, stream);                            CUERR
    
    int chunk = max(min(1<<10, N-M+1), 2*M);
    int lower = 0, upper = chunk;
    
    gpu_cdtw<euclidean>(Subject, AvgS, StdS, Cdtw, M, Indices, chunk, W, 
                        true, stream);                                    CUERR
    reduce(Cdtw, Best_cdtw, chunk, cub::Min(), stream);                   CUERR
    cudaMemcpyAsync(&best_cdtw, Best_cdtw, sizeof(value_t), 
                    cudaMemcpyDeviceToHost, stream);                      CUERR
    cudaMemcpyAsync(&max_kim, Lb_kim+chunk-1, sizeof(value_t), 
                    cudaMemcpyDeviceToHost, stream);                      CUERR

    if (best_cdtw < max_kim) {
        report_result<euclidean>(Cdtw, Indices, upper, stream);           CUERR
    } else {
        for (lower = upper; lower < N-M+1; lower += chunk) {
            
            // update upper index and double chunk size
            chunk = min(2*chunk, 1<<16);
            upper = min(upper+chunk, N-M+1);
            
            // calculate Lb_Keogh on current chunk
            lb_keogh<euclidean>(Subject, AvgS, StdS, Lb_keogh+lower, 
                                M, Indices+lower, upper-lower,stream);    CUERR
            
            index_t length;
            threshold(Lb_keogh+lower, Indices+lower, upper-lower, 
                      &length, best_cdtw, stream);
            
            if (length == 0)
                continue;
            
            gpu_cdtw<euclidean>(Subject, AvgS, StdS, Cdtw+lower, M, 
                                Indices+lower, length, W, true, stream);  CUERR
            reduce(Cdtw+lower, Best_cdtw, length, cub::Min(), stream);    CUERR
    
            cudaMemcpyAsync(&tmp, Best_cdtw, sizeof(value_t), 
                            cudaMemcpyDeviceToHost, stream);              CUERR
            cudaMemcpyAsync(&max_kim, Lb_kim+upper-1, sizeof(value_t), 
                            cudaMemcpyDeviceToHost, stream);              CUERR
            
            best_cdtw = min(tmp, best_cdtw);
            //std::cout << length*100.0/(upper-lower) << "\t" << best_cdtw << "\t" << max_kim << std::endl;
            if (best_cdtw < max_kim) {
                break;
            }
        }
    }
    
    report_result<euclidean>(Cdtw, Indices, upper, stream);               CUERR

    std::cout << (best_cdtw < max_kim) << std::endl;

    cudaFree(Best_cdtw);                                                  CUERR
    cudaFree(Lb_kim);                                                     CUERR
    cudaFree(Lb_keogh);                                                   CUERR
    cudaFree(Cdtw);                                                       CUERR
    cudaFree(Indices);                                                    CUERR
}

#endif


