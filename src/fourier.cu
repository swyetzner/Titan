//
// Created by sw3390 on 9/7/20.
//

#include "fourier.h"

Fourier::Fourier(double uf, double lf, int b) {
    upperFreq = uf;
    lowerFreq = lf;
    bands = b;
    n = 0;
    ts = 0;
    last_recorded = 0;

    massComplexArray = nullptr;
}

void Fourier::operator=(CUDA_FOURIER & fourier) {
    upperFreq = fourier.upperFreq;
    lowerFreq = fourier.lowerFreq;
    bands = fourier.bands;
    n = fourier.n;
    massComplexArray = fourier.massComplexArray;
}

CUDA_FOURIER::CUDA_FOURIER(Fourier &fourier) {
    upperFreq = fourier.upperFreq;
    lowerFreq = fourier.lowerFreq;
    bands = fourier.bands;
    n = fourier.n;
    massComplexArray = fourier.massComplexArray;
}
