//
//  vec.hpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
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

class Vec {
public:

    double data[3] = { 0 }; // initialize data to 0

    CUDA_CALLABLE_MEMBER Vec();
    CUDA_CALLABLE_MEMBER Vec(const Vec & v);
    CUDA_CALLABLE_MEMBER Vec(double x, double y, double z);
    
    CUDA_CALLABLE_MEMBER Vec & operator=(const Vec & v);
    CUDA_CALLABLE_MEMBER Vec & operator+=(const Vec & v);
    CUDA_CALLABLE_MEMBER Vec & operator-=(const Vec & v);

    CUDA_DEVICE void atomicVecAdd(const Vec & v);
    CUDA_DEVICE void atomicVecExch(const Vec &v);

    CUDA_CALLABLE_MEMBER Vec & operator+=(double x);
    CUDA_CALLABLE_MEMBER Vec & operator-=(double x);
    CUDA_CALLABLE_MEMBER Vec & operator*=(double x);
    CUDA_CALLABLE_MEMBER Vec & operator/=(double x);

    CUDA_CALLABLE_MEMBER Vec operator-() const;
    CUDA_CALLABLE_MEMBER double & operator [] (int n);
    CUDA_CALLABLE_MEMBER const double & operator [] (int n) const;

    CUDA_CALLABLE_MEMBER friend Vec operator+(const Vec & v1, const Vec & v2);
    CUDA_CALLABLE_MEMBER friend Vec operator-(const Vec & v1, const Vec & v2);
    CUDA_CALLABLE_MEMBER friend Vec operator*(const double x, const Vec & v);
    CUDA_CALLABLE_MEMBER friend Vec operator*(const Vec & v, const double x);
    CUDA_CALLABLE_MEMBER friend bool operator==(const Vec & v1, const Vec & v2);
    CUDA_CALLABLE_MEMBER friend bool operator<(const Vec &v1, const Vec &v2);
    CUDA_CALLABLE_MEMBER friend bool operator>(const Vec &v1, const Vec &v2);
    CUDA_CALLABLE_MEMBER friend Vec operator*(const Vec & v1, const Vec & v2);
    CUDA_CALLABLE_MEMBER friend Vec operator/(const Vec & v, const double x);
    CUDA_CALLABLE_MEMBER friend Vec operator/(const Vec & v1, const Vec & v2);

    friend std::ostream & operator << (std::ostream & strm, const Vec & v);

    CUDA_CALLABLE_MEMBER void print();
    CUDA_CALLABLE_MEMBER double norm() const;
    CUDA_CALLABLE_MEMBER double sum() const;
    CUDA_CALLABLE_MEMBER Vec normalized() const;

private:
};

CUDA_CALLABLE_MEMBER double dot(const Vec & a, const Vec & b);
/*
CUDA_CALLABLE_MEMBER double dot(const Vec & a, const Vec & b) {
    return (a * b).sum();
}*/

template <class T>
CUDA_CALLABLE_MEMBER T cross(const T &v1, const T &v2) {
    return T(v1[1] * v2[2] - v1[2] * v2[1], v2[0] * v1[2] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}



#endif

