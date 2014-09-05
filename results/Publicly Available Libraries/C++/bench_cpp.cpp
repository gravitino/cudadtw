#include<iostream>
#include<fstream>
#include<limits>
#include <ctime>
#include <stdlib.h> 

double dtw(const double * query, const double * subject, const int M) {

    double * penalty = new double[(M+1)*(M+1)];
    penalty[0] = 0.0;
    
    for (int j = 1; j < M+1; ++j)
        penalty[j] = std::numeric_limits<double>::infinity();
    
    for (int i = 1; i < M+1; ++i)
        penalty[i*(M+1)] = std::numeric_limits<double>::infinity();

    for (int i = 1; i < M+1; ++i)
        for(int j = 1; j < M+1; ++j) {
            
            const double value = query[i-1]-subject[j-1];
        
            penalty[i*(M+1)+j] = value*value + 
                                 std::min<double>(penalty[(i-1)*(M+1)+j-1],
                                 std::min<double>(penalty[(i-1)*(M+1)+j],
                                                  penalty[i*(M+1)+j-1]));
        }
        
    const double measure = penalty[(M+1)*(M+1)-1];
    delete [] penalty;
    
    return measure;
}

void sdtw(const double * query, const double * subject, 
            const int M, const int N) {
    
    for (int k = 0; k < N-M+1; ++k) {
        dtw(query, subject+k, M);
    }
}

int main(int argc, char* argv[]){

    if (argc != 5){
        std::cout << "call" << argv[0] 
                  << " query.bin subject.bin M N" << std::endl; 
        return 1;
    }
    
    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    
    double * query = new double[M];
    double * subject = new double[N];
    
    std::ifstream qfile(argv[1], std::ios::binary|std::ios::in);
    qfile.read((char *) &query[0], sizeof(double)*M);

    std::ifstream sfile(argv[2], std::ios::binary|std::ios::in);
    sfile.read((char *) &subject[0], sizeof(double)*N);
    
    clock_t begin = clock();
    sdtw(query, subject, M, N);
    clock_t end = clock();
    std::cout << (double(end - begin) / CLOCKS_PER_SEC) << std::endl;
    
    delete [] query;
    delete [] subject;
}
