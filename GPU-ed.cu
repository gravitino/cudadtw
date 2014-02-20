#include <iostream>
#include <fstream>


#include "include/ed.cuh"
#include "include/cuda_def.cuh"


int main(int argc, char* argv[]) {

     if (argc != 5){
        std::cout << "call" << argv[0] 
                  << " query.bin subject.bin M N" << std::endl; 
        return 1;
    }

    cudaSetDevice(0);                                                     CUERR
    cudaDeviceReset();                                                    CUERR

    double *zquery = NULL, *subject = NULL,
           *Subject = NULL, *AvgS = NULL, *StdS = NULL; 

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);

    
    std::cout << "\n= info =====================================" << std::endl;
    std::cout << "|Query| = " << M << "\t"
              << "|Subject| = " << N << "\t" << std::endl;

    // host side memory
    cudaMallocHost(&zquery, sizeof(double)*M);                            CUERR
    cudaMallocHost(&subject, sizeof(double)*N);                           CUERR

    // device side memory
    cudaMalloc(&Subject, sizeof(double)*N);                               CUERR
    cudaMalloc(&AvgS, sizeof(double)*(N-M+1));                            CUERR
    cudaMalloc(&StdS, sizeof(double)*(N-M+1));                            CUERR
    
    // timer events
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\n= loading data =============================" << std::endl;
    
    cudaEventRecord(start, 0);
    
    // read query from file
    std::ifstream qfile(argv[1], std::ios::binary|std::ios::in);
    qfile.read((char *) &zquery[0], sizeof(double)*M);

    // read subject from file
    std::ifstream sfile(argv[2], std::ios::binary|std::ios::in);
    sfile.read((char *) &subject[0], sizeof(double)*N);

    // z-normalize query and envelope
    znormalize(zquery, M);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to load data: " << time << std::endl;
    
    cudaEventRecord(start, 0);
    
    // copy subject to gpu
    cudaMemcpy(Subject, subject, sizeof(double)*N, 
               cudaMemcpyHostToDevice);                                   CUERR
    // copy query to constant memory
    cudaMemcpyToSymbol(::Czquery, zquery, sizeof(double)*M);              CUERR

    // calculate windowed average and standard deviation of Subject
    avg_std<double>(Subject, AvgS, StdS, M, N);

    // average not needed anymore
    cudaFree(AvgS);                                                       CUERR
    
    // calculate best z-normalized Euclidean match
    calculate_ed(Subject, StdS, M, N);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Miliseconds to find best match: " << time << std::endl;
}

