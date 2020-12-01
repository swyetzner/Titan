//
// Created by sw3390 on 9/7/20.
//
#include <assert.h>
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

Fourier::~Fourier() {
    for (int i = 0; i < bands; i++) {
        delete[] massComplexArray[i];
        delete[] modeShapes[i];
    }

    delete[] massComplexArray;
    delete[] modeShapes;

    delete[] expTerms;
    delete[] frequencies;
}

void Fourier::operator=(CUDA_FOURIER & fourier) {
    upperFreq = fourier.upperFreq;
    lowerFreq = fourier.lowerFreq;
    bands = fourier.bands;
    n = fourier.n;
    n_count = fourier.n_count;
    nmasses = fourier.nmasses;
    expTerms = fourier.expTerms;
    expTerms = fourier.expTerms;

    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < nmasses; j++) {
            massComplexArray[i][j] = fourier.massComplexArray[i * nmasses + j];
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
    expTerms = fourier.expTerms;

    if (fourier.massComplexArray) {
        massComplexArray = new ComplexVec[bands * nmasses];
        for (int i = 0; i < bands; i++) {
            for (int j = 0; j < nmasses; j++) {
                massComplexArray[i * nmasses + j] = fourier.massComplexArray[i][j];
            }
        }
    }
}
