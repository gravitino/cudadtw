#include <iostream>
#include <fstream>

#include "include/ldtw.cuh"

int main(int argc, char* argv[]) {

    if (argc != 6){
        std::cout << "call" << argv[0] 
                  << " query.bin subject.bin M N P" << std::endl; 
        return 1;
    }

    cudaSetDevice(0);                                                     CUERR
    cudaDeviceReset();                                                    CUERR

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int W = M*(atoi(argv[5])*0.01);

    std::cout << " M " << M << " N " << N << " W " << W << std::endl;

    float *q = NULL, *s = NULL, *r = NULL, *Q = NULL, *S = NULL, *R = NULL, 
          *avgs = NULL, *stds = NULL, *AvgS = NULL, *StdS = NULL;

    // host memory
    cudaMallocHost(&q, sizeof(float)*M);                                  CUERR
    cudaMallocHost(&s, sizeof(float)*N);                                  CUERR
    cudaMallocHost(&r, sizeof(float)*(N-M+1));                            CUERR
    cudaMallocHost(&avgs, sizeof(float)*(N-M+1));                         CUERR
    cudaMallocHost(&stds, sizeof(float)*(N-M+1));                         CUERR

    // device memory
    cudaMalloc(&Q, sizeof(float)*M);                                      CUERR
    cudaMalloc(&S, sizeof(float)*N);                                      CUERR
    cudaMalloc(&R, sizeof(float)*(N-M+1));                                CUERR
    cudaMalloc(&AvgS, sizeof(float)*(N-M+1));                             CUERR
    cudaMalloc(&StdS, sizeof(float)*(N-M+1));                             CUERR

    // timer events
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\n= loading data =============================" << std::endl;
    
    cudaEventRecord(start, 0);
    
    // read query from file
    std::ifstream qfile(argv[1], std::ios::binary|std::ios::in);
    qfile.read((char *) &q[0], sizeof(float)*M);

    // read subject from file
    std::ifstream sfile(argv[2], std::ios::binary|std::ios::in);
    sfile.read((char *) &s[0], sizeof(float)*N);

    // z-normalize query and envelope
    znormalize(q, M);
   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to load data: " << time << std::endl;

    cudaMemcpy(Q, q, sizeof(float)*M, cudaMemcpyHostToDevice);            CUERR
    cudaMemcpy(S, s, sizeof(float)*N, cudaMemcpyHostToDevice);            CUERR

    // bind query and subject texture
    cudaBindTexture(0, TS, S, N*sizeof(float));                           CUERR
    cudaBindTexture(0, TQ, Q, M*sizeof(float));                           CUERR
    
    // calculate windowed average and standard deviation of Subject
    avg_std<double>(S, AvgS, StdS, M, N);
   
    std::cout << "\n= aligning==================================" << std::endl;

    cudaEventRecord(start, 0);
    
    // use this for alignment without z-normalization
    //tsh2_ldtw<true><<<SDIV(N-M+1, 1024), 1024, (1024+2*M)*sizeof(float)>>> 
    //               (Q, M, S, N, R, W);                                  CUERR
 
    ztsh2_ldtw<true><<<SDIV(N-M+1, 1024), 1024, (1024+2*M)*sizeof(float)>>> 
              (Q, M, S, N, AvgS, StdS, R, W);                             CUERR
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to find best alignment on the GPU: " 
              << time << std::endl;
    
    cudaMemcpy(r, R, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost);      CUERR

    // consistency check
    determine_min<true>(r, N-M+1);
    
    cudaMemcpy(avgs, AvgS, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost);CUERR
    cudaMemcpy(stds, StdS, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost);CUERR

    cudaDeviceSynchronize();
    
    // reset result vector
    for (int k = 0; k < N-M+1; ++k)
        r[k] = 0;
    
    cudaEventRecord(start, 0);
    
    omp_z_stream_ldtw<true>(q, M, s, N, avgs, stds, r, W);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to find best alignment on all cores: " 
              << time << std::endl;

    // consistency check
    determine_min<true>(r, N-M+1);

              
    // reset result vector
    for (int k = 0; k < N-M+1; ++k)
        r[k] = 0;
    
    cudaEventRecord(start, 0);
    
    z_stream_ldtw<true>(q, M, s, N, avgs, stds, r, W);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to find best alignment on a single core: " 
              << time << std::endl;
    
    // consistency check
    determine_min<true>(r, N-M+1);

}
