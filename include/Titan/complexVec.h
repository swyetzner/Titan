//
//  complexVec.h
//  CUDA Physics
//
//  Created by Salvy on 10/12/20.
//  Copyright © 2020 Salvy. All rights reserved.
//

#ifndef VEC_H
#define VEC_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <iostream>
#include <cmath>

#include <cuComplex.h>

class ComplexVec {
public:

    cuDoubleComplex data[3] = { 0 }; // initialize data to 0

    CUDA_CALLABLE_MEMBER ComplexVec();
    CUDA_CALLABLE_MEMBER ComplexVec(const ComplexVec & v);
    CUDA_CALLABLE_MEMBER ComplexVec(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex z);
    CUDA_CALLABLE_MEMBER ComplexVec & operator=(const ComplexVec & v);

    CUDA_CALLABLE_MEMBER ComplexVec & operator+=(const ComplexVec & v);
    CUDA_CALLABLE_MEMBER ComplexVec & operator-=(const ComplexVec & v);
    CUDA_CALLABLE_MEMBER ComplexVec & operator*=(const ComplexVec & v);
    CUDA_CALLABLE_MEMBER ComplexVec & operator/=(const ComplexVec & v);

    CUDA_CALLABLE_MEMBER ComplexVec & operator+=(double x);
    CUDA_CALLABLE_MEMBER ComplexVec & operator-=(double x);
    CUDA_CALLABLE_MEMBER ComplexVec & operator*=(double x);
    CUDA_CALLABLE_MEMBER ComplexVec & operator/=(double x);

    //CUDA_CALLABLE_MEMBER ComplexVec operator-() const; // returns the negative -z = -a -bi
    CUDA_CALLABLE_MEMBER cuDoubleComplex & operator [] (int n);
    CUDA_CALLABLE_MEMBER const cuDoubleComplex & operator [] (int n) const;
    CUDA_CALLABLE_MEMBER friend bool operator==(const ComplexVec & v1, const ComplexVec & v2);

    CUDA_CALLABLE_MEMBER friend ComplexVec operator+(const ComplexVec & v1, const ComplexVec & v2);
    CUDA_CALLABLE_MEMBER friend ComplexVec operator-(const ComplexVec & v1, const ComplexVec & v2);
    CUDA_CALLABLE_MEMBER friend ComplexVec operator*(const ComplexVec & v1, const ComplexVec & v2);
    CUDA_CALLABLE_MEMBER friend ComplexVec operator/(const ComplexVec & v1, const ComplexVec & v2);

    friend std::ostream & operator << (std::ostream & strm, const ComplexVec & v);

    CUDA_CALLABLE_MEMBER void print();
    //CUDA_CALLABLE_MEMBER double norm() const;
    CUDA_CALLABLE_MEMBER cuDoubleComplex sum() const;

private:
};


CUDA_CALLABLE_MEMBER double dot(const Vec & a, const Vec & b);
CUDA_CALLABLE_MEMBER Vec cross(const Vec &v1, const Vec &v2);

#endif