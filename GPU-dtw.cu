#include <iostream>
#include <fstream>

#include "include/cuda_def.cuh"
#include "include/cdtw.cuh"

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

    float *zlower = NULL, *zupper = NULL, *zquery = NULL, *subject = NULL,
          *Subject = NULL, *AvgS = NULL, *StdS = NULL; 

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int W = M*(atoi(argv[5])*0.01);
    
    std::cout << "\n= info =====================================" << std::endl;
    std::cout << "|Query| = " << M << "\t"
              << "|Subject| = " << N << "\t"
              << "window = " << W << std::endl;

    // host side memory
    cudaMallocHost(&zlower, sizeof(float)*M);                             CUERR
    cudaMallocHost(&zupper, sizeof(float)*M);                             CUERR
    cudaMallocHost(&zquery, sizeof(float)*M);                             CUERR
    cudaMallocHost(&subject, sizeof(float)*N);                            CUERR

    // device side memory
    cudaMalloc(&Subject, sizeof(float)*N);                                CUERR
    cudaMalloc(&AvgS, sizeof(float)*(N-M+1));                             CUERR
    cudaMalloc(&StdS, sizeof(float)*(N-M+1));                             CUERR

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
    envelope(zquery, W, zlower, zupper, M);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to load data: " << time << std::endl;
    
    cudaEventRecord(start, 0);
    
    // copy subject to gpu
    cudaMemcpy(Subject, subject, sizeof(float)*N, 
               cudaMemcpyHostToDevice);                                   CUERR
    // copy query and associated envelopes to constant memory
    cudaMemcpyToSymbol(::Czlower, zlower, sizeof(float)*M);               CUERR
    cudaMemcpyToSymbol(::Czupper, zupper, sizeof(float)*M);               CUERR
    cudaMemcpyToSymbol(::Czquery, zquery, sizeof(float)*M);               CUERR

    // calculate windowed average and standard deviation of Subject
    avg_std<double>(Subject, AvgS, StdS, M, N);
    std::cout << "\n= pruning ==================================" << std::endl;
    
    prune_cdtw<EUCLIDEAN>(Subject, AvgS, StdS, M, N, W);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to find best match: " << time << std::endl;
}

