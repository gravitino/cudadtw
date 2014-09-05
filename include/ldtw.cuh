#ifndef LDTW_CUH
#define LDTW_CUH

#include <omp.h>
#include "cuda_def.cuh"
#include "stats.cuh"

#define INFTY (1E10)

// textures for query and subject
texture<float, 1, cudaReadModeElementType> TS;
texture<float, 1, cudaReadModeElementType> TQ;

//////////////////////////////////////////////////////////////////////////////
// Host LDTW portion
//////////////////////////////////////////////////////////////////////////////

// serial cLDTW without z-normalization
template <bool Euclidean, class index_t, class value_t>
float serial_ldtw(const value_t * query, const value_t * subject, 
                  const index_t M, const index_t W) {
    
    // initialize indices and residues
    index_t i = 0, j = 0;
    value_t d = query[i] - subject[j];
            d = Euclidean ? d*d : d < 0 ? -d : d;

    while (i+1 != M || j+1 != M) {
        
        // reset distances
        value_t diag = INFTY,
                rght = INFTY,
                down = INFTY;
    
        // relax diagonal cell
        if (i+1 < M && j+1 < M) {
            diag = query[i+1] - subject[j+1];;
            diag = Euclidean ? diag*diag : diag < 0 ? -diag : diag;
        }
        
        // relax cell below
        if (i+1 < M && abs(i-j+1) <= W) {
            down = query[i+1] - subject[j];
            down = Euclidean ? down*down : down < 0 ? -down : down;
        }
        
        // relax right cell
        if (j+1 < M && abs(j-i+1) <= W) {
            rght = query[i] - subject[j+1];
            rght = Euclidean ? rght*rght : rght < 0 ? -rght : rght;
        }
        
        const bool diagSdown = diag < down;
        const bool diagSrght = diag < rght;
        const bool downSrght = down < rght;
        
        if(diagSdown && diagSrght) {
            d += diag;
            i++;
            j++;
        } else if (!diagSdown && downSrght) {
            d += down;
            i++;
        } else {
            d += rght;
            j++;
        }
    }

    return d;
}

// serial cLDTW with z-normalization
template <bool Euclidean, class index_t, class value_t>
float z_serial_ldtw(const value_t * query, const value_t * subject, 
                    const value_t avg, const value_t std,
                    const index_t M, const index_t W) {
    
    // initialize indices and residues
    index_t i = 0, j = 0;
    value_t d = query[i] - (subject[j]-avg)/std;
            d = Euclidean ? d*d : d < 0 ? -d : d;
    
    while (i+1 != M || j+1 != M) {
        
        // reset distances
        value_t diag = INFTY,
                rght = INFTY,
                down = INFTY;
        
        // relax diagonal cell
        if (i+1 < M && j+1 < M) {
            diag = query[i+1]-(subject[j+1]-avg)/std;
            diag = Euclidean ? diag*diag : diag < 0 ? -diag : diag;
        }
        
        // relax cell below
        if (i+1 < M && abs(i-j+1) <= W) {
            down = query[i+1]-(subject[j]-avg)/std;
            down = Euclidean ? down*down : down < 0 ? -down : down;
        }
        
        // relax right cell
        if (j+1 < M && abs(j-i+1) <= W) {
            rght = query[i]-(subject[j+1]-avg)/std;;
            rght = Euclidean ? rght*rght : rght < 0 ? -rght : rght;
        }
        
        const bool diagSdown = diag < down;
        const bool diagSrght = diag < rght;
        const bool downSrght = down < rght;
        
        if(diagSdown && diagSrght) {
            d += diag;
            i++;
            j++;
        } else if (!diagSdown && downSrght) {
            d += down;
            i++;
        } else {
            d += rght;
            j++;
        }
    }

    return d;
}

// stream cLDTW without z-normalization
template <bool Euclidean, class index_t, class value_t> __host__
void stream_ldtw(const value_t * query,   const index_t M,
                 const value_t * subject, const index_t N,
                 value_t * result, const index_t W) {

    for (int k = 0; k < N-M+1; ++k)
        result[k] = serial_ldtw<Euclidean>(query, subject+k, M, W);

}

// stream cLDTW without z-normalization (openmp)
template <bool Euclidean, class index_t, class value_t> __host__
void omp_stream_ldtw(const value_t * query,   const index_t M,
                     const value_t * subject, const index_t N,
                     value_t * result, const index_t W) {

    # pragma omp parallel for
    for (int k = 0; k < N-M+1; ++k)
        result[k] = serial_ldtw<Euclidean>(query, subject+k, M, W);
}

// stream cLDTW wit z-normalization
template <bool Euclidean, class index_t, class value_t> __host__
void z_stream_ldtw(const value_t * query,   const index_t M,
                   const value_t * subject, const index_t N,
                   const value_t * avgs,    const value_t * stds,
                   value_t * result, const index_t W) {

    for (int k = 0; k < N-M+1; ++k)
        result[k] = z_serial_ldtw<Euclidean>
                    (query, subject+k, avgs[k], stds[k], M, W);
}

// stream cLDTW wit z-normalization (openmp)
template <bool Euclidean, class index_t, class value_t> __host__
void omp_z_stream_ldtw(const value_t * query,   const index_t M,
                       const value_t * subject, const index_t N,
                       const value_t * avgs,    const value_t * stds,
                       value_t * result, const index_t W) {

    # pragma omp parallel for
    for (int k = 0; k < N-M+1; ++k)
        result[k] = z_serial_ldtw<Euclidean>
                    (query, subject+k, avgs[k], stds[k], M, W);
}

// consistency check
template <bool Euclidean, class index_t, class value_t>
void determine_min(const value_t * result, const index_t L) {
    
    value_t bsf = INFTY;
    index_t bsf_index = 0;
    
    for (int i = 0; i < L; ++i)
        if (result[i] < bsf) {
            bsf = result[i];
            bsf_index = i;
        }
    
    std::cout << "best alignment found at index " << bsf_index 
              << " with distance " << (Euclidean ? sqrt(bsf) : bsf) 
              << std::endl << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
// GPU LDTW portion
//////////////////////////////////////////////////////////////////////////////

// stream cLDTW without z-normalization
template <bool Euclidean, class index_t, class value_t> __global__
void tsh2_ldtw(const value_t * query,   const index_t M,
               const value_t * subject, const index_t N,
               value_t * result, const index_t W) {

    // first global threadId in block and actual global threadId
    int base = blockDim.x*blockIdx.x; 
    int thid = base+threadIdx.x;

    // load query and subject to shared memory
    extern __shared__ float cache[];
    float * shs = cache, * shq = cache+M+blockDim.x;
    
    for (int i = threadIdx.x ; i < blockDim.x+M; i += blockDim.x)
        shs[i] = tex1Dfetch(TS, i+base);

    for (int i = threadIdx.x ; i < M; i += blockDim.x)
        shq[i] = query[i];

    __syncthreads();

    // get the work done
    if (thid < N-M+1) {
        
        // initialize the relaxation procedure
        value_t d = shq[0] - shs[threadIdx.x];
        d = Euclidean ? d*d : d < 0 ? -d : d;
        index_t i = 0, j = 0;
    
        // cache
        value_t diag, rght, down, sp0, sp1, qp0, qp1, bsf;
        bool valid_i, valid_j, cond_down, cond_rght;
        
        // relax 2M-2 times (first relaxation in initialization)
        for (int k = 0; k < 2*M-2; ++k) {
        
            // reset distances
            diag = INFTY;
            rght = INFTY;
            down = INFTY;
            bsf  = INFTY;
            
            // calculate conditions for the window and global borders
            valid_i = i+1 < M;
            valid_j = j+1 < M;
            cond_down = abs(i-j+1) <= W;
            cond_rght = abs(j-i+1) <= W;
        
            // read the involved values once
            sp0 = shs[j+threadIdx.x];
            sp1 = shs[(j+1+threadIdx.x)*valid_j];
            qp0 = shq[i];
            qp1 = shq[(i+1)*valid_i];
        
             // relax diagonal cell
            diag = qp1 - sp1;
            diag = Euclidean ? diag*diag : diag < 0 ? -diag : diag;
            diag = valid_i && valid_j ? diag : INFTY;
        
             // relax cell below
            down = qp1 - sp0;
            down = Euclidean ? down*down : down < 0 ? -down : down;
            down = valid_i && cond_down ? down : INFTY;
        
             // relax right cell
            rght = qp0 - sp1;
            rght = Euclidean ? rght*rght : rght < 0 ? -rght : rght;
            rght = valid_j && cond_rght ? rght : INFTY;

            // determine argmin
            const bool diagSdown = diag < down;
            const bool diagSrght = diag < rght;
            const bool downSrght = down < rght;
        
            if(diagSdown && diagSrght) {
                bsf = diag;
                i = min(i+1, M-1);
                j = min(j+1, M-1);
            } else if (!diagSdown && downSrght) {
                bsf = down;
                i = min(i+1, M-1);
            } else {
                bsf = rght;
                j = min(j+1, M-1);
            }
            
            // update distance
            d += (valid_i || valid_j) ? bsf : 0;
        }
        
        result[thid] = d;
    }
}

// stream cLDTW with z-normalization
template <bool Euclidean, class index_t, class value_t> __global__
void ztsh2_ldtw(const value_t * query,   const index_t M,
                const value_t * subject, const index_t N,
                const value_t * avgs,    const value_t * stds,
                value_t * result, const index_t W) {

    // first global threadId in block and actual global threadId
    int base = blockDim.x*blockIdx.x; 
    int thid = base+threadIdx.x;

    // load query and subject to shared memory
    extern __shared__ float cache[];
    float * shs = cache, * shq = cache+M+blockDim.x;
    
    for (int i = threadIdx.x ; i < blockDim.x+M; i += blockDim.x)
        shs[i] = tex1Dfetch(TS, i+base);

    for (int i = threadIdx.x ; i < M; i += blockDim.x)
        shq[i] = tex1Dfetch(TQ, i);
    
    __syncthreads();

    // get the work done
    if (thid < N-M+1) {
        
        // read average and standard deviation (query already z-normalized)
        value_t avg = avgs[thid];
        value_t std = stds[thid];
        
        // initialize the relaxation procedure
        value_t d = shq[0] - (shs[threadIdx.x]-avg)/std;
        d = Euclidean ? d*d : d < 0 ? -d : d;
        index_t i = 0, j = 0;
    
        // cache
        value_t diag, rght, down, sp0, sp1, qp0, qp1, bsf;
        bool valid_i, valid_j, cond_down, cond_rght;
        
        // relax 2M-2 times (first relaxation in initialization)
        for (int k = 0; k < 2*M-2; ++k) {
        
            // reset distances
            diag = INFTY;
            rght = INFTY;
            down = INFTY;
            bsf  = INFTY;
            
            // calculate conditions for the window and global borders
            valid_i = i+1 < M;
            valid_j = j+1 < M;
            cond_down = abs(i-j+1) <= W;
            cond_rght = abs(j-i+1) <= W;
        
            // read the involved values once
            sp0 = (shs[j+threadIdx.x]-avg)/std;
            sp1 = (shs[(j+1+threadIdx.x)*valid_j]-avg)/std;
            qp0 = shq[i];
            qp1 = shq[(i+1)*valid_i];
        
            // relax diagonal cell
            diag = qp1 - sp1;
            diag = Euclidean ? diag*diag : diag < 0 ? -diag : diag;
            diag = valid_i && valid_j ? diag : INFTY;
        
            // relax cell below
            down = qp1 - sp0;
            down = Euclidean ? down*down : down < 0 ? -down : down;
            down = valid_i && cond_down ? down : INFTY;
        
            // relax right cell
            rght = qp0 - sp1;
            rght = Euclidean ? rght*rght : rght < 0 ? -rght : rght;
            rght = valid_j && cond_rght ? rght : INFTY;

            // determine argmin
            const bool diagSdown = diag < down;
            const bool diagSrght = diag < rght;
            const bool downSrght = down < rght;
        
            if(diagSdown && diagSrght) {
                bsf = diag;
                i = min(i+1, M-1);
                j = min(j+1, M-1);
            } else if (!diagSdown && downSrght) {
                bsf = down;
                i = min(i+1, M-1);
            } else {
                bsf = rght;
                j = min(j+1, M-1);
            }
            
            // update distance
            d += (valid_i || valid_j) ? bsf : 0;
        }
        
        result[thid] = d;
    }
}

#endif
