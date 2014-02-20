#include <iostream>
#include <fstream>
#include <algorithm>

#include "include/cdtw.cuh"
#include "include/stats.cuh"

#define EUCLIDEAN (true)
#define MANHATTAN (false)

int main(int argc, char* argv[]) {


     if (argc != 6){
        std::cout << "call" << argv[0] 
                  << " query.bin subject.bin M N P" << std::endl; 
        return 1;
    }

    cudaSetDevice(0);                                                     CUERR
    cudaDeviceReset();                                                    CUERR

    float *zquery = NULL, *subject = NULL, 
          *avgS = NULL, *stdS = NULL, *cdtw = NULL, 
          *Subject = NULL, *AvgS = NULL, *StdS = NULL, *Cdtw = NULL; 

    int *indices = NULL, *Indices = NULL;

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int W = M*(atoi(argv[5])*0.01);
    
    std::cout << "\n= info =====================================" << std::endl;
    std::cout << "|Query| = " << M << "\t"
              << "|Subject| = " << N << "\t"
              << "window = " << W << std::endl;

    // host side memory
    cudaMallocHost(&zquery, sizeof(float)*M);                             CUERR
    cudaMallocHost(&subject, sizeof(float)*N);                            CUERR
    cudaMallocHost(&avgS, sizeof(float)*(N-M+1));                         CUERR
    cudaMallocHost(&stdS, sizeof(float)*(N-M+1));                         CUERR
    cudaMallocHost(&cdtw, sizeof(float)*(N-M+1));                         CUERR
    cudaMallocHost(&indices, sizeof(int)*(N-M+1));                        CUERR

    // device side memory
    cudaMalloc(&Subject, sizeof(float)*N);                                CUERR
    cudaMalloc(&AvgS, sizeof(float)*(N-M+1));                             CUERR
    cudaMalloc(&StdS, sizeof(float)*(N-M+1));                             CUERR
    cudaMalloc(&Cdtw, sizeof(float)*(N-M+1));                             CUERR
    cudaMalloc(&Indices, sizeof(int)*(N-M+1));                            CUERR

    // bind subject texture
    cudaBindTexture(0, Tsubject, Subject, N*sizeof(float));               CUERR
    
    // timer events
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\n= loading data =============================" << std::endl;
    
    cudaEventRecord(start, 0);
    
    // read query from file
    std::ifstream qfile(argv[1], std::ios::binary|std::ios::in);
    qfile.read((char *) &zquery[0], sizeof(float)*M);

    // read subject from file
    std::ifstream sfile(argv[2], std::ios::binary|std::ios::in);
    sfile.read((char *) &subject[0], sizeof(float)*N);

    // z-normalize query and envelope
    znormalize(zquery, M);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to load data: " << time << std::endl;
    
    // create a random set of indices
    for (int i = 0; i < N-M+1; ++i)
        indices[i] = i;
    std::random_shuffle(indices, indices+N-M+1);

    cudaMemcpy(Indices, indices, sizeof(int)*(N-M+1), 
               cudaMemcpyHostToDevice);                                   CUERR

    // copy subject to gpu
    cudaMemcpy(Subject, subject, sizeof(float)*N, 
               cudaMemcpyHostToDevice);                                   CUERR
    // copy query and associated envelopes to constant memory
    cudaMemcpyToSymbol(::Czquery, zquery, sizeof(float)*M);               CUERR

    // calculate windowed average and standard deviation of Subject
    avg_std<double>(Subject, AvgS, StdS, M, N);

    cudaEventRecord(start, 0);
    
    // call gpu dtw
    gpu_cdtw<EUCLIDEAN>(Subject, AvgS, StdS, Cdtw, M, Indices, N-M+1, W); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU BLOCK: " << time << std::endl;

    // save result for consistency check
    float *gpucdtw = NULL;
    cudaMallocHost(&gpucdtw, sizeof(float)*(N-M+1));                      CUERR
    cudaMemcpy(gpucdtw, Cdtw, sizeof(float)*(N-M+1), 
               cudaMemcpyDeviceToHost);                                   CUERR

    cudaEventRecord(start, 0);
    
    // call gpu dtw
    gpu_cdtw<EUCLIDEAN>(Subject, AvgS, StdS, Cdtw, M, Indices, N-M+1, W, false); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU THRD: " << time << std::endl;

    // copy statistics back to host
    cudaMemcpy(avgS, AvgS, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost);CUERR
    cudaMemcpy(stdS, StdS, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost);CUERR

    cudaEventRecord(start, 0);
    
    // call openmp dtw
    cpu_cdtw <EUCLIDEAN>
    (zquery, subject, avgS, stdS, cdtw, M, indices, N-M+1, W);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "CPU OMP: " << time << std::endl;
    
    cudaEventRecord(start, 0);
    
    // call sequential dtw
    cpu_cdtw <EUCLIDEAN>
    (zquery, subject, avgS, stdS, cdtw, M, indices, N-M+1, W, false);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "CPU SEQ: " << time << std::endl;
    
    for (int i = 0; i < N-M+1; ++i)
        if(fabs(gpucdtw[i]-cdtw[i]) > 1E-4)
            std::cout << i << "\t" << gpucdtw[i] << "\t" << cdtw[i] << std::endl;
}
