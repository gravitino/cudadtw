#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <typeinfo>
#include <numeric>
#include <vector>
#include <cmath>
#include <map>
#include <time.h>

#include "mkl_dfti.h"


#define REG(x) ((x) == 0 ? 1 : x)

//////////////////////////////////////////////////////////////////////////////
// functors
//////////////////////////////////////////////////////////////////////////////

template <class T> struct divide_by {
  const T y;  
  divide_by(const T& y_) : y(y_) {};
  T operator() (const T& x) const {return x/y;}
};

template <class T> struct multiply_by {
  const T y;  
  multiply_by(const T& y_) : y(y_) {};
  T operator() (const T& x) const {return x*y;}
};

template <class T> struct square {
  T operator() (const T& x) const {return x*x;}
  T operator() (const T& x, const T& y) const {return x+y*y;}
};

template <class T> struct compose_std {
  T operator() (const T& x, const T& y) const {return REG(sqrt(y-x*x));}
};

//////////////////////////////////////////////////////////////////////////////
// windowed prefix sums and time series statistics
//////////////////////////////////////////////////////////////////////////////

template <class Iterator>
int windowed_prefix(Iterator ib, Iterator ie, Iterator rb, size_t w) {

    // deduce floating point type from Iterator
    typedef typename std::iterator_traits<Iterator>::value_type ftype;

    // get length of input
    size_t L = ie-ib;

    // calculate out of place prefix sum
    std::vector<ftype> prefix(L+1, 0);
    std::partial_sum(ib, ie, prefix.begin()+1);
   
    // calculate windowed difference
    std::transform(prefix.begin()+w, prefix.end(),
                   prefix.begin(), rb, std::minus<ftype>());


    // successful
    return 0;
}

template <class Iterator, class F>
int transformed_windowed_prefix(Iterator ib, Iterator ie, Iterator rb, 
                                size_t w, F fn) {

    // deduce floating point type from Iterator
    typedef typename std::iterator_traits<Iterator>::value_type ftype;

    // get length of input
    size_t L = ie-ib;

    // calculate out of place prefix sum
    std::vector<ftype> prefix(L+1, 0);
    std::copy(ib, ie, prefix.begin()+1);
    std::transform(prefix.begin()+1, prefix.begin()+L+1, prefix.begin()+1, fn);
    std::partial_sum(prefix.begin()+1, prefix.begin()+L+1, prefix.begin()+1);    
   
    // calculate windowed difference
    std::transform(prefix.begin()+w, prefix.end(),
                   prefix.begin(), rb, std::minus<ftype>());


    // successful
    return 0;
}

template <class Iterator>
int mu_sigma(Iterator ib, Iterator ie, Iterator mb, Iterator sb, size_t w) {
    
    // deduce floating point type from Iterator
    typedef typename std::iterator_traits<Iterator>::value_type ftype;
    
    // get length of input
    size_t L = ie-ib;
    
    // get moving average
    windowed_prefix(ib, ie, mb, w);
    std::transform(mb, mb+L-w+1, mb, divide_by<ftype>(w));

    // get first part of variance
    transformed_windowed_prefix(ib, ie, sb, w, square<ftype>());
    std::transform(sb, sb+L-w+1, sb, divide_by<ftype>(w));
    
    // compose standard deviation
    std::transform(mb, mb+L-w+1, sb, sb, compose_std<ftype>());
    
    // successful
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
// correlation methods
//////////////////////////////////////////////////////////////////////////////

template <class Iterator>
int correlate(Iterator qb, Iterator qe, Iterator sb, Iterator rb) {
    
    // deduce floating point type from Iterator
    typedef typename std::iterator_traits<Iterator>::value_type ftype;

    if (typeid(ftype) != typeid(double) && typeid(ftype) != typeid(float)) {
        std::cerr << "ERROR: series type must be float or double! (CRITICAL)" 
                  << std::endl;
        // unsuccessful (type error)
        return 1;
    }
 
    // determine the length of input arrays and allocate fourier space
    const size_t L = qe-qb;
    const size_t F = (L/2+1)*2;
    std::vector<ftype> fftq(F, 0.0);
    std::vector<ftype> ffts(F, 0.0);
    
    // prepare forward transform
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG status = 0;
    
    if (typeid(ftype) == typeid(double))
        status |= DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, L);
    else
        status |= DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, L);
    
    // transform query and subject
    status |= DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status |= DftiCommitDescriptor(handle);
    status |= DftiComputeForward(handle, &qb[0], &fftq[0]);
    status |= DftiComputeForward(handle, &sb[0], &ffts[0]);
    
    // multipy fourier transforms
    for (Iterator pq = fftq.begin(), ps = ffts.begin(); 
        pq < fftq.end(); pq +=2, ps+=2){
        
        // get values and multiply conjugated
        const ftype a = *pq, b = *(pq+1), c = *ps, d = *(ps+1);
        const ftype real = a*c + b*d, imag = a*d - b*c;
        
        // write down result
        *(pq) = real/L; *(pq+1) = imag/L;
    }
    
    // write result and free handle
    status |= DftiComputeBackward(handle, &fftq[0], &rb[0]);
    status |= DftiFreeDescriptor(&handle);
    
    // status
    if (status != 0)
        std::cerr << "ERROR: error during fourier transform! (CRITICAL)"
                  << std::endl;
                    
    return static_cast<int>(status);
}

//////////////////////////////////////////////////////////////////////////////
// matching methods
//////////////////////////////////////////////////////////////////////////////

template <class Iterator>
int znorm_local_ed(Iterator qb, Iterator qe, 
                   Iterator sb, Iterator se, Iterator rb) {

    // deduce floating point type from Iterator
    typedef typename std::iterator_traits<Iterator>::value_type ftype;

    if (typeid(ftype) != typeid(double) && typeid(ftype) != typeid(float)) {
        std::cerr << "ERROR: series type must be float or double! (CRITICAL)" 
                  << std::endl;
        // unsuccessful (type error)
        return 1;
    }
    
    if (typeid(ftype) == typeid(float))
        std::cerr << "WARNING: single precision may be insufficient "
                  << "for long queries or subject time series!" << std::endl;

    // sizes of query and subject
    size_t M = qe-qb, N = se-sb;
    
    if (M > N) {
        std::cerr << "ERROR: query longer than subject! (CRITICAL)" 
                  << std::endl; 
        // unsuccessful (length error)
        return 2;
    }

    // remember status during computation
    int status = 0;

    // calculate statistics of query
    const ftype avgQ = std::accumulate(qb, qe, static_cast<ftype>(0))/M;
    const ftype stdQ = sqrt(std::accumulate(qb, qe, static_cast<ftype>(0), 
                                            square<ftype>())/M-avgQ*avgQ);

    // calculate statistics of subject
    std::vector<ftype> avgS(N-M+1);
    std::vector<ftype> stdS(N-M+1);
    status |= mu_sigma(sb, se, avgS.begin(), stdS.begin(), M);
    
    // calculate correlation terms between query and subject
    std::vector<ftype> corr(N, 0);
    std::copy(qb, qe, corr.begin());
    status |= correlate(corr.begin(), corr.end(), sb, corr.begin());
    
    // calculate the final result: (corr-M*muQ*muS)/(M*stdQ*stdS)
    std::transform(avgS.begin(), avgS.end(), avgS.begin(), 
                   multiply_by<ftype>(avgQ*M));
    std::transform(stdS.begin(), stdS.end(), stdS.begin(), 
                   multiply_by<ftype>(stdQ*M));
    std::transform(avgS.begin(), avgS.end(), corr.begin(), corr.begin(), 
                   std::minus<ftype>());
    std::transform(corr.begin(), corr.end()-M+1, stdS.begin(), corr.begin(), 
                   std::divides<ftype>());
    
    // copy result
    std::copy(corr.begin(), corr.end()-M+1, rb);
    
    // status
    return status;
}

int main(int argc, char* argv[]) {

    if (argc != 5){
        std::cout << "call" << argv[0] 
                  << " query.bin subject.bin M N" << std::endl; 
        return 1;
    }

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);

    std::cout << "\n= info =====================================" << std::endl;
    std::cout << "|Query| = " << M << "\t"
              << "|Subject| = " << N << "\t" << std::endl;

    std::vector<double> query(M);
    std::vector<double> subject(N);
    std::vector<double> result (N);

    std::cout << "\n= loading data =============================" << std::endl;

     // read query from file
    std::ifstream qfile(argv[1], std::ios::binary|std::ios::in);
    qfile.read((char *) &query[0], sizeof(double)*M);

    // read subject from file
    std::ifstream sfile(argv[2], std::ios::binary|std::ios::in);
    sfile.read((char *) &subject[0], sizeof(double)*N);
    
    double t1 = clock();
    
    znorm_local_ed(query.begin(), query.end(), 
                   subject.begin(), subject.end(), result.begin());

    double bsf = 100000;
    int bsf_index= - 1;

    for (int i = 0; i < N-M+1; ++i)
        if(bsf > result[i]) {
            bsf = result[i];
            bsf_index=i;
        }

    double t2 = clock();
    
    std::cout << "\n= result ===================================" << std::endl;
    std::cout << "distance: " << sqrt(2*M*(1+bsf)) << std::endl;
    std::cout << "location: " << bsf_index << std::endl;

    std::cout << "Miliseconds to find best match: " << (t2-t1)/CLOCKS_PER_SEC*1000 
              << std::endl;

}




