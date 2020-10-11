//
//  vec.cpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include "vec.h"

#if __CUDA_ARCH__ < 600
__device__ double atomicDoubleAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicDoubleExch(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

CUDA_DEVICE void Vec::atomicVecAdd(const Vec & v) {
atomicDoubleAdd(&data[0], (double) v.data[0]);
atomicDoubleAdd(&data[1], (double) v.data[1]);
atomicDoubleAdd(&data[2], (double) v.data[2]);
}

CUDA_DEVICE void Vec::atomicVecExch(const Vec &v) {
    atomicDoubleExch(&data[0], (double) v.data[0]);
    atomicDoubleExch(&data[1], (double) v.data[1]);
    atomicDoubleExch(&data[2], (double) v.data[2]);
}

Vec::Vec() {
        data[0] = 0;
        data[1] = 0;
    	data[2] = 0;
} // default

Vec::Vec(const Vec & v) {
    data[0] = v.data[0];
    data[1] = v.data[1];
    data[2] = v.data[2];
} // copy constructor

Vec::Vec(double x, double y, double z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
} // initialization from x, y, and z values

Vec & Vec::operator=(const Vec & v) {
    if (this == &v) {
        return *this;
    }

    data[0] = v.data[0];
    data[1] = v.data[1];
    data[2] = v.data[2];

    return *this;
}

Vec & Vec::operator+=(const Vec & v) {
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return *this;
}

Vec & Vec::operator-=(const Vec & v) {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return *this;
}

Vec Vec::operator-() const{
    return Vec(-data[0], -data[1], -data[2]);
}

double & Vec::operator [] (int n) {
    if (n < 0 || n >= 3) {
        printf("%s\n", "Out of bounds!");
        return data[0];
    } else {
        return data[n];
    }
}

const double & Vec::operator [] (int n) const {
    if (n < 0 || n >= 3) {
        printf("%s\n", "Out of bounds!");
        return data[0];
    } else {
        return data[n];
    }
}

Vec operator+(const Vec & v1, const Vec & v2) {
    return Vec(v1.data[0] + v2.data[0], v1.data[1] + v2.data[1], v1.data[2] + v2.data[2]);
}

Vec operator-(const Vec & v1, const Vec & v2) {
    return Vec(v1.data[0] - v2.data[0], v1.data[1] - v2.data[1], v1.data[2] - v2.data[2]);
}

Vec operator*(const double x, const Vec & v) {
    return Vec(v.data[0] * x, v.data[1] * x, v.data[2] * x);
}

Vec operator*(const Vec & v, const double x) {
    return x * v;
}

bool operator==(const Vec & v1, const Vec & v2) {
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
}

bool operator<(const Vec &v1, const Vec &v2) {
    return ((v1[0] < v2[0]) || (v1[0] <= v2[0] && v1[1] < v2[1])
            || (v1[0] <= v2[0] && v1[1] <= v2[1] && v1[2] < v2[2]));
}

bool operator>(const Vec &v1, const Vec &v2) {
    return !(v1<v2);
}

Vec operator*(const Vec & v1, const Vec & v2) {
    return Vec(v1.data[0] * v2.data[0], v1.data[1] * v2.data[1], v1.data[2] * v2.data[2]);
} // Multiplies two Vecs (elementwise)

Vec operator/(const Vec & v, const double x) {
    return Vec(v.data[0] / x, v.data[1] / x, v.data[2] / x);
} //  vector over double

Vec operator/(const Vec & v1, const Vec & v2) {
    return Vec(v1.data[0] / v2.data[0], v1.data[1] / v2.data[1], v1.data[2] / v2.data[2]);
} // divides two Vecs (elementwise)

std::ostream & operator << (std::ostream & strm, const Vec & v) {
    return strm << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
} // print

void Vec::print() {
    printf("(%3f, %3f, %3f)\n", data[0], data[1], data[2]);
}

double Vec::norm() const {
    return sqrt(pow(data[0], 2) + pow(data[1], 2) + pow(data[2], 2));
} // gives vector norm

double Vec::sum() const {
    return data[0] + data[1] + data[2];
} // sums all components of the vector

Vec Vec::normalized() const {
    double l = norm();
    return l > 0 ? (*this)/l : (*this);
}

CUDA_CALLABLE_MEMBER double dot(const Vec & a, const Vec & b) {
    return (a * b).sum();
}

CUDA_CALLABLE_MEMBER Vec cross(const Vec &v1, const Vec &v2) {
    return Vec(v1[1] * v2[2] - v1[2] * v2[1], v2[0] * v1[2] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}


















