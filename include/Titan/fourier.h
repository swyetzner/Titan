//
// Created by sw3390 on 9/7/20.
//

#ifndef TITAN_FOURIER_H
#define TITAN_FOURIER_H

#include "vec.h"
#include "complexVec.h"
#include <complex>

class Fourier;
struct CUDA_FOURIER;

struct CUDA_FOURIER {
    CUDA_FOURIER() = default;
    CUDA_FOURIER(Fourier & fourier);

    double upperFreq;
    double lowerFreq;
    int bands;
    int n;
    ComplexVec *expTerms;
    double *frequencies;
    Vec ** modeShapes;
    ComplexVec ** massComplexArray;
};

class Fourier {
public:
    // Properties
    double upperFreq;
    double lowerFreq;
    int bands;
    int n; // Derived
    double ts; // Derived
    double last_recorded;

    ComplexVec *expTerms;
    double *frequencies;
    Vec ** modeShapes;
    ComplexVec ** massComplexArray;

    Fourier() = default;
    Fourier(double uf, double lf, int b);

private:
    void operator=(CUDA_FOURIER & fourier);
};

#endif //TITAN_FOURIER_H
