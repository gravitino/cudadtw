#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include<list>             // envelope

///////////////////////////////////////////////////////////////////////////////
// CPU Lemire's streaming min-max filter
///////////////////////////////////////////////////////////////////////////////

// host side envelope
// Lemire: "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", 
// Pattern Recognition 42(9), 2009.
template <class value_t, class index_t> __host__
void envelope(value_t* series, index_t W, 
              value_t* L, value_t* U, index_t M) {
    
    // Daniel Lemire's windowed min-max algorithm in O(3n):
    std::list<index_t> u = std::list<index_t>();
    std::list<index_t> l = std::list<index_t>();

    u.push_back(0);
    l.push_back(0);

    for (int i = 1; i < M; ++i) {

        if (i > W) {

            U[i-W-1] = series[u.front()];
            L[i-W-1] = series[l.front()];
        }
        
        if (series[i] > series[i-1]) {
            
            u.pop_back();
            while (!u.empty() && series[i] > series[u.back()])
                u.pop_back();
        } else {

            l.pop_back();
            while (!l.empty() && series[i] < series[l.back()])
                l.pop_back();
        }
        
        u.push_back(i);
        l.push_back(i);
        
        if (i == 2*W+1+u.front())
            u.pop_front();
        else if (i == 2*W+1+l.front())
            l.pop_front();
    }

    for (int i = M; i < M+W+1; ++i) {

        U[i-W-1] = series[u.front()];
        L[i-W-1] = series[l.front()];

        if (i-u.front() >= 2*W+1)
            u.pop_front();

        if (i-l.front() >= 2*W+1)
            l.pop_front();
    }
}

///////////////////////////////////////////////////////////////////////////////
// GPU Lb_Keogh
///////////////////////////////////////////////////////////////////////////////

template <bool euclidean, class value_t, class index_t> __global__
void shared_lb_keogh(value_t* Subject, value_t* AvgS, value_t* StdS, 
                     value_t* Lb_keogh, index_t M, index_t N) {
    
    int indx = blockDim.x*blockIdx.x + threadIdx.x;

    // shared memory
    extern __shared__ value_t cache[];
    
    // load subject including halo pixels into shared memory
    for (int offset = 0; offset < blockDim.x+M; offset += blockDim.x)
        if (offset+threadIdx.x < blockDim.x+M && offset+indx < N)
            cache[offset+threadIdx.x] = Subject[offset+indx];
    __syncthreads();
    
    // calculate LB_Keogh
    if (indx < N-M+1) {
    
        // obtain statistics
        value_t residues= 0;
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];

        for (int i = 0; i < M; ++i) {
        
            // differences to envelopes
            value_t value = (cache[threadIdx.x+i]-avg)/std;
            value_t lower = value-Czlower[i];
            value_t upper = value-Czupper[i];
            
            // Euclidean or Manhattan distance?
            if (euclidean)
                residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
            else
                residues += upper*(upper > 0) - lower*(lower<0);
        }
        
        Lb_keogh[indx] = residues;
    }
}

template <bool euclidean, class value_t, class index_t> __global__
void random_lb_keogh(value_t* Subject, value_t* AvgS, value_t* StdS, 
                     value_t* Lb_keogh, index_t M, index_t *Indices, 
                     index_t indices_l) {
    
    int thid = blockDim.x*blockIdx.x + threadIdx.x;

    // calculate LB_Keogh
    if (thid < indices_l) {
    
        int indx = Indices[thid];
    
        // obtain statistics
        value_t residues= 0;
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];

        for (int i = 0; i < M; ++i) {
        
            // differences to envelopes
            value_t value = (Subject[indx+i]-avg)/std;
            value_t lower = value-Czlower[i];
            value_t upper = value-Czupper[i];
            
            // Euclidean or Manhattan distance?
            if (euclidean)
                residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
            else
                residues += upper*(upper > 0) - lower*(lower<0);
        }
        
        Lb_keogh[thid] = residues;
    }
}

///////////////////////////////////////////////////////////////////////////////
// GPU LB_Kim
///////////////////////////////////////////////////////////////////////////////

#define lp(x,y) ((x) ? (y)*(y) : (y) < 0 ? -(y) : (y))

template <bool euclidean, class value_t, class index_t> __global__
void random_lb_kim(value_t* Subject, value_t* AvgS, value_t* StdS, 
                   value_t* Lb_kim, index_t M, 
                   index_t *Indices, index_t indices_l) {

    int seid = blockIdx.x*blockDim.x+threadIdx.x;
    
    // registers for 10-point DTW
    value_t q0, q1, q2, q3, q4, s0, s1, s2, s3, s4, mem, 
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
    
    if (seid < indices_l) {
        
        //actual index
        int indx = Indices[seid];
        
        // get statistics for given index
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];
        
        // read query
        q0 = Czquery[0];
        q1 = Czquery[1];
        q2 = Czquery[2];
        q3 = Czquery[3];
        q4 = Czquery[4];
        
        // read subject and z-normalize it
        s0 = (Subject[indx]-avg)/std;
        s1 = (Subject[indx+1]-avg)/std;
        s2 = (Subject[indx+2]-avg)/std;
        s3 = (Subject[indx+3]-avg)/std;
        s4 = (Subject[indx+4]-avg)/std;

        // relax the first row
        p0 = lp(euclidean, q0-s0);
        p1 = lp(euclidean, q0-s1) + p0;
        p2 = lp(euclidean, q0-s2) + p1;
        p3 = lp(euclidean, q0-s3) + p2;
        p4 = lp(euclidean, q0-s4) + p3;
        mem = p4;
        
        // relax the second row
        p5 = lp(euclidean, q1-s0) + p0;
        p6 = lp(euclidean, q1-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q1-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q1-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q1-s4) + min(p8, min(p3, p4));
        mem = min(mem, p9);
        
        // relax the third row
        p0 = lp(euclidean, q2-s0) + p5;
        p1 = lp(euclidean, q2-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q2-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q2-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q2-s4) + min(p3, min(p8, p9));
        mem = min(mem, p4);
        
        // relax the fourth row
        p5 = lp(euclidean, q3-s0) + p0;
        p6 = lp(euclidean, q3-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q3-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q3-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q3-s4) + min(p8, min(p3, p4));
        mem = min(mem, p9);
        
        // relax the fith row
        p0 = lp(euclidean, q4-s0) + p5;
        p1 = lp(euclidean, q4-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q4-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q4-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q4-s4) + min(p3, min(p8, p9));
        mem = min(mem, min(p0, min(p1, min(p2, min(p3, p4)))));
        
        
        // now do the same for the end of the window
        
        // read query
        q0 = Czquery[M-5];
        q1 = Czquery[M-4];
        q2 = Czquery[M-3];
        q3 = Czquery[M-2];
        q4 = Czquery[M-1];
        
        // read subject and z-normalize it
        s0 = (Subject[indx+M-5]-avg)/std;
        s1 = (Subject[indx+M-4]-avg)/std;
        s2 = (Subject[indx+M-3]-avg)/std;
        s3 = (Subject[indx+M-2]-avg)/std;
        s4 = (Subject[indx+M-1]-avg)/std;

        // relax the first row
        p0 = lp(euclidean, q0-s0);
        p1 = lp(euclidean, q0-s1);
        p2 = lp(euclidean, q0-s2);
        p3 = lp(euclidean, q0-s3);
        p4 = lp(euclidean, q0-s4);
        
        // relax the second row
        p5 = lp(euclidean, q1-s0);
        p6 = lp(euclidean, q1-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q1-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q1-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q1-s4) + min(p8, min(p3, p4));
        
        // relax the third row
        p0 = lp(euclidean, q2-s0);
        p1 = lp(euclidean, q2-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q2-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q2-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q2-s4) + min(p3, min(p8, p9));
        
        // relax the fourth row
        p5 = lp(euclidean, q3-s0);
        p6 = lp(euclidean, q3-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q3-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q3-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q3-s4) + min(p8, min(p3, p4));
       
        // relax the fith row
        p0 = lp(euclidean, q4-s0);
        p1 = lp(euclidean, q4-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q4-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q4-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q4-s4) + min(p3, min(p8, p9));
                
        Lb_kim[seid] = mem+p4;
    }

}

template <bool euclidean, class value_t, class index_t> __global__
void register_lb_kim(value_t* Subject, value_t* AvgS, value_t* StdS, 
                   value_t* Lb_kim, index_t M, index_t N) {

    int indx = blockIdx.x*blockDim.x+threadIdx.x;
    
    // registers for 10-point DTW
    value_t q0, q1, q2, q3, q4, s0, s1, s2, s3, s4, mem, 
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
    
    if (indx < N-M+1) {
        

        // get statistics for given index
        value_t avg = AvgS[indx];
        value_t std = StdS[indx];
        
        // read query
        q0 = Czquery[0];
        q1 = Czquery[1];
        q2 = Czquery[2];
        q3 = Czquery[3];
        q4 = Czquery[4];
        
        // read subject and z-normalize it
        s0 = (Subject[indx]-avg)/std;
        s1 = (Subject[indx+1]-avg)/std;
        s2 = (Subject[indx+2]-avg)/std;
        s3 = (Subject[indx+3]-avg)/std;
        s4 = (Subject[indx+4]-avg)/std;

        // relax the first row
        p0 = lp(euclidean, q0-s0);
        p1 = lp(euclidean, q0-s1) + p0;
        p2 = lp(euclidean, q0-s2) + p1;
        p3 = lp(euclidean, q0-s3) + p2;
        p4 = lp(euclidean, q0-s4) + p3;
        mem = p4;
        
        // relax the second row
        p5 = lp(euclidean, q1-s0) + p0;
        p6 = lp(euclidean, q1-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q1-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q1-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q1-s4) + min(p8, min(p3, p4));
        mem = min(mem, p9);
        
        // relax the third row
        p0 = lp(euclidean, q2-s0) + p5;
        p1 = lp(euclidean, q2-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q2-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q2-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q2-s4) + min(p3, min(p8, p9));
        mem = min(mem, p4);
        
        // relax the fourth row
        p5 = lp(euclidean, q3-s0) + p0;
        p6 = lp(euclidean, q3-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q3-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q3-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q3-s4) + min(p8, min(p3, p4));
        mem = min(mem, p9);
        
        // relax the fith row
        p0 = lp(euclidean, q4-s0) + p5;
        p1 = lp(euclidean, q4-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q4-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q4-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q4-s4) + min(p3, min(p8, p9));
        mem = min(mem, min(p0, min(p1, min(p2, min(p3, p4)))));
        
        // now do the same for the end of the window
        
        // read query
        q0 = Czquery[M-5];
        q1 = Czquery[M-4];
        q2 = Czquery[M-3];
        q3 = Czquery[M-2];
        q4 = Czquery[M-1];
        
        // read subject and z-normalize it
        s0 = (Subject[indx+M-5]-avg)/std;
        s1 = (Subject[indx+M-4]-avg)/std;
        s2 = (Subject[indx+M-3]-avg)/std;
        s3 = (Subject[indx+M-2]-avg)/std;
        s4 = (Subject[indx+M-1]-avg)/std;

        // relax the first row
        p0 = lp(euclidean, q0-s0);
        p1 = lp(euclidean, q0-s1);
        p2 = lp(euclidean, q0-s2);
        p3 = lp(euclidean, q0-s3);
        p4 = lp(euclidean, q0-s4);
        
        // relax the second row
        p5 = lp(euclidean, q1-s0);
        p6 = lp(euclidean, q1-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q1-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q1-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q1-s4) + min(p8, min(p3, p4));
        
        // relax the third row
        p0 = lp(euclidean, q2-s0);
        p1 = lp(euclidean, q2-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q2-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q2-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q2-s4) + min(p3, min(p8, p9));
        
        // relax the fourth row
        p5 = lp(euclidean, q3-s0);
        p6 = lp(euclidean, q3-s1) + min(p5, min(p0, p1));
        p7 = lp(euclidean, q3-s2) + min(p6, min(p1, p2));
        p8 = lp(euclidean, q3-s3) + min(p7, min(p2, p3));
        p9 = lp(euclidean, q3-s4) + min(p8, min(p3, p4));
       
        // relax the fith row
        p0 = lp(euclidean, q4-s0);
        p1 = lp(euclidean, q4-s1) + min(p0, min(p5, p6));
        p2 = lp(euclidean, q4-s2) + min(p1, min(p6, p7));
        p3 = lp(euclidean, q4-s3) + min(p2, min(p7, p8));
        p4 = lp(euclidean, q4-s4) + min(p3, min(p8, p9));
                
        Lb_kim[indx] = mem+p4;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Wrappers
///////////////////////////////////////////////////////////////////////////////

template <bool euclidean, class value_t, class index_t> __host__
void lb_kim(value_t* Subject, value_t* AvgS, value_t* StdS, value_t* Lb_kim, 
            index_t M, index_t *Indices, index_t indices_l,
            cudaStream_t stream=0) {

    if (M < 11) {
        std::cout << "ERROR: LB_Kim is a hardcoded 10 point unconstrained DTW."
                  << "Chose a query length of at least 11!" << std::endl;
        return;
    }
    
    random_lb_kim<euclidean> <<<SDIV(indices_l, 1024), 1024, 0, stream>>>
    (Subject, AvgS, StdS, Lb_kim, M, Indices, indices_l);                 CUERR
}


template <bool euclidean, class value_t, class index_t> __host__
void lb_kim(value_t* Subject, value_t* AvgS, value_t* StdS, value_t* Lb_kim, 
            index_t M, index_t N, cudaStream_t stream=0) {

    if (M < 11) {
        std::cout << "ERROR: LB_Kim is a hardcoded 10 point unconstrained DTW."
                  << "Chose a query length of at least 11!" << std::endl;
        return;
    }
    
    register_lb_kim<euclidean> <<<SDIV(N-M+1, 1024), 1024, 0, stream>>>
    (Subject, AvgS, StdS, Lb_kim, M, N);                                  CUERR
}

template <bool euclidean, class value_t, class index_t> __host__
void lb_keogh(value_t* Subject, value_t* AvgS, value_t* StdS, value_t* Lb_keogh, 
              index_t M, index_t *Indices, index_t indices_l, 
              cudaStream_t stream=0) {

    int blockdim = 1024;
    int griddim = SDIV(indices_l, blockdim);

    random_lb_keogh<euclidean> <<<griddim, blockdim, 0, stream>>>
    (Subject, AvgS, StdS, Lb_keogh, M, Indices, indices_l);               CUERR
}

template <bool euclidean, class value_t, class index_t> __host__
void lb_keogh(value_t* Subject, value_t* AvgS, value_t* StdS, value_t* Lb_keogh,
              index_t M, index_t N, cudaStream_t stream) {

    int blockdim = 1024;
    int griddim = SDIV(N-M+1, blockdim);
    int memory = (M+blockdim)*sizeof(value_t);
    
    shared_lb_keogh<euclidean> <<<griddim, blockdim, memory, stream>>>
    (Subject, AvgS, StdS, Lb_keogh, M, N);                                CUERR
}


#endif
