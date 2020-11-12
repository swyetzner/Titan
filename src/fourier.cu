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

    expTerms = nullptr;
    massComplexArray = nullptr;
}

void Fourier::operator=(CUDA_FOURIER & fourier) {
    upperFreq = fourier.upperFreq;
    lowerFreq = fourier.lowerFreq;
    bands = fourier.bands;
    n = fourier.n;
    n_count = fourier.n_count;
    nmasses = fourier.nmasses;
    expTerms = fourier.expTerms;
    frequencies = fourier.frequencies;
    modeShapes = fourier.modeShapes;
    expTerms = fourier.expTerms;

    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < nmasses; j++) {
            massComplexArray[i][j] = fourier.massComplexArray[i*bands + j];
        }
    }
}

CUDA_FOURIER::CUDA_FOURIER(Fourier &fourier) {
    upperFreq = fourier.upperFreq;
    lowerFreq = fourier.lowerFreq;
    bands = fourier.bands;
    n = fourier.n;
    n_count = fourier.n_count;
    nmasses = fourier.nmasses;
    expTerms = fourier.expTerms;
    frequencies = fourier.frequencies;
    modeShapes = fourier.modeShapes;
    expTerms = fourier.expTerms;

    massComplexArray = new ComplexVec[bands * nmasses];
    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < nmasses; j++) {
            massComplexArray[i*bands + j] = fourier.massComplexArray[i][j];
        }
    }
}
