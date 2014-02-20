#include <iostream>
#include <algorithm>
#include "include/cub_util.cuh"

#define L (100)

int main () {

    int *indices = NULL, *Indices = NULL;
    float *values = NULL, *Values = NULL;
    
    cudaMallocHost(&indices, sizeof(int)*L);
    cudaMallocHost(&values, sizeof(float)*L);
    
    cudaMalloc(&Indices, sizeof(int)*L);
    cudaMalloc(&Values, sizeof(float)*L);

    for (int i = 0; i < L; ++i)
        indices[i] = values[i] = i;
    std::random_shuffle(values, values+L);

    for (int i = 0; i < L;  ++i)
        std::cout << indices[i] << "\t" << values[i] << std::endl;
    std::cout << "=================================" << std::endl;
    
    cudaMemcpy(Indices, indices, sizeof(int)*L, cudaMemcpyHostToDevice);
    cudaMemcpy(Values, values, sizeof(float)*L, cudaMemcpyHostToDevice);
    
    int length;
    threshold(Values, Indices, L, &length, 22.0f);
    
    cudaMemcpy(indices, Indices, sizeof(int)*length, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < length; ++i)
        std::cout << i << "\t" << indices[i] << std::endl;

}
