//
// Created by sw3390 on 9/7/20.
//

#ifndef TITAN_FOURIER_H
#define TITAN_FOURIER_H

#include "vec.h"
#include "complexVec.h"
#include <complex>
#include <vector>

class Fourier;
struct CUDA_FOURIER;
struct CUDA_FOURIER_POINTERS;

struct CUDA_FOURIER {
    CUDA_FOURIER() = default;
    CUDA_FOURIER(Fourier & fourier);
    ~CUDA_FOURIER();

    double upperFreq;
    double lowerFreq;
    int bands;
    int n;
    int n_count;
    int nmasses;
    ComplexVec *expTerms;
    ComplexVec * massComplexArray; // 1D array
};

struct CUDA_FOURIER_POINTERS {
    CUDA_FOURIER_POINTERS() {
        d_fourier = nullptr;
        d_massComplexArray = nullptr;
        d_expTerms = nullptr;
    }

    CUDA_FOURIER * d_fourier;
    ComplexVec * d_massComplexArray;
    ComplexVec * d_expTerms;
};

class Fourier {
public:
    // Properties
    double upperFreq;
    double lowerFreq;
    int bands;
    int n; // Derived
    int n_count;
    int nmasses;
    double ts; // Derived
    double last_recorded;

    ComplexVec *expTerms = nullptr;
    double *frequencies = nullptr;
    Vec ** modeShapes = nullptr;
    ComplexVec ** massComplexArray = nullptr;

    Fourier() = default;
    Fourier(double uf, double lf, int b);
    ~Fourier();

    void operator=(CUDA_FOURIER & fourier);
};

#endif //TITAN_FOURIER_H
