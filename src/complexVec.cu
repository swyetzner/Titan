//
//  complexVec.cpp
//  CUDA Physics
//
//  Created by Salvy Cavicchio on 10/11/20.
//  Copyright © 2020 Salvy Cavicchio. All rights reserved.
//

#include "complexVec.h"

// COMPLEX VECTOR

ComplexVec::ComplexVec() {
    data[0] = make_cuDoubleComplex(0, 0);
    data[1] = make_cuDoubleComplex(0, 0);
   	data[2] = make_cuDoubleComplex(0, 0);
} // default

ComplexVec::ComplexVec(const ComplexVec & v) {
    data[0] = v.data[0];
    data[1] = v.data[1];
    data[2] = v.data[2];
} // copy constructor

ComplexVec::ComplexVec(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
} // initialization from x, y, and z values

ComplexVec & ComplexVec::operator=(const ComplexVec & v) {
    if (this == &v) {
        return *this;
    }

    data[0] = v.data[0];
    data[1] = v.data[1];
    data[2] = v.data[2];

    return *this;
}

ComplexVec & ComplexVec::operator+=(const ComplexVec & v) {
	data[0] = cuCadd(data[0],v.data[0]);
    data[1] = cuCadd(data[1],v.data[1]);
    data[2] = cuCadd(data[2],v.data[2]);
    return *this;
}

ComplexVec & ComplexVec::operator+=(const Vec & v) {
    data[0] = cuCadd(data[0],make_cuDoubleComplex(v[0], 0));
    data[1] = cuCadd(data[1],make_cuDoubleComplex(v[1], 0));
    data[2] = cuCadd(data[2],make_cuDoubleComplex(v[2], 0));
    return *this;
}

ComplexVec & ComplexVec::operator-=(const ComplexVec & v) {
	data[0] = cuCsub(data[0],v.data[0]);
    data[1] = cuCsub(data[1],v.data[1]);
    data[2] = cuCsub(data[2],v.data[2]);
    return *this;
}

ComplexVec & ComplexVec::operator-=(const Vec & v) {
    data[0] = cuCsub(data[0], make_cuDoubleComplex(v.data[0], 0));
    data[1] = cuCsub(data[1], make_cuDoubleComplex(v.data[1], 0));
    data[2] = cuCsub(data[2], make_cuDoubleComplex(v.data[2], 0));
    return *this;
}

ComplexVec & ComplexVec::operator*=(const ComplexVec & v) {
	data[0] = cuCmul(data[0],v.data[0]);
    data[1] = cuCmul(data[1],v.data[1]);
    data[2] = cuCmul(data[2],v.data[2]);
    return *this;
}

ComplexVec & ComplexVec::operator*=(const Vec & v) {
    data[0] = cuCmul(data[0],make_cuDoubleComplex(v[0], 0));
    data[1] = cuCmul(data[1],make_cuDoubleComplex(v[1], 0));
    data[2] = cuCmul(data[2],make_cuDoubleComplex(v[2], 0));
    return *this;
}

ComplexVec & ComplexVec::operator/=(const ComplexVec & v) {
	data[0] = cuCdiv(data[0],v.data[0]);
    data[1] = cuCdiv(data[1],v.data[1]);
    data[2] = cuCdiv(data[2],v.data[2]);
    return *this;
}

ComplexVec & ComplexVec::operator+=(double x) {
	data[0] = cuCadd(data[0],make_cuDoubleComplex(x, 0));
	data[1] = cuCadd(data[1],make_cuDoubleComplex(x, 0));
	data[2] = cuCadd(data[2],make_cuDoubleComplex(x, 0));
	return *this;
}

ComplexVec & ComplexVec::operator-=(double x) {
	data[0] = cuCsub(data[0],make_cuDoubleComplex(x, 0));
	data[1] = cuCsub(data[1],make_cuDoubleComplex(x, 0));
	data[2] = cuCsub(data[2],make_cuDoubleComplex(x, 0));
	return *this;
}

ComplexVec & ComplexVec::operator*=(double x) {
	data[0] = cuCmul(data[0],make_cuDoubleComplex(x, 0));
	data[1] = cuCmul(data[1],make_cuDoubleComplex(x, 0));
	data[2] = cuCmul(data[2],make_cuDoubleComplex(x, 0));
	return *this;
}

ComplexVec & ComplexVec::operator/=(double x) {
	data[0] = cuCdiv(data[0],make_cuDoubleComplex(x, 0));
	data[1] = cuCdiv(data[1],make_cuDoubleComplex(x, 0));
	data[2] = cuCdiv(data[2],make_cuDoubleComplex(x, 0));
	return *this;
}

//CUDA_CALLABLE_MEMBER ComplexVec operator-() const; // returns the negative -z = -a -bi
cuDoubleComplex & ComplexVec::operator [] (int n) {
	if (n < 0 || n >= 3) {
        printf("%s\n", "Out of bounds!");
        return data[0];
    } else {
        return data[n];
    }
}

const cuDoubleComplex & ComplexVec::operator [] (int n) const {
	if (n < 0 || n >= 3) {
        printf("%s\n", "Out of bounds!");
        return data[0];
    } else {
        return data[n];
    }
}

bool operator==(const ComplexVec & v1, const ComplexVec & v2) {
	return ((v1[0].x == v2[0].x && v1[1].x == v2[1].x && v1[2].x == v2[2].x) && (v1[0].y == v2[0].y && v1[1].y == v2[1].y && v1[2].y == v2[2].y));
}

ComplexVec operator+(const ComplexVec & v1, const ComplexVec & v2) {
	return ComplexVec(v1+v2);
}

ComplexVec operator+(const ComplexVec & v1, const Vec & v2) {
    return ComplexVec(v1+v2);
}

ComplexVec operator-(const ComplexVec & v1, const ComplexVec & v2) {
	return ComplexVec(v1-v2);
}

ComplexVec operator*(const ComplexVec & v1, const ComplexVec & v2) {
	return ComplexVec(v1*v2);
}

ComplexVec operator*(const ComplexVec & v1, const Vec & v2) {
    return ComplexVec(v1*v2);
}

ComplexVec operator/(const ComplexVec & v1, const ComplexVec & v2) {
	return ComplexVec(v1/v2);
}

std::ostream & operator << (std::ostream & strm, const ComplexVec & v) {
	return strm << "(" << v[0].x << " + " << v[0].y << "i" << ", " << v[1].x << " + " << v[0].y << "i" << ", " << v[2].x << " + " << v[0].y << "i)";
}

void ComplexVec::print() {
	printf("(%3f + %3fi, %3f + %3fi, %3f + %3fi)\n", data[0].x, data[0].y, data[1].x, data[1].y, data[2].x, data[2].y);
}

cuDoubleComplex ComplexVec::sum() const {
	return cuCadd(cuCadd(data[0],data[1]),data[2]);
}

ComplexVec ComplexVec::exp() {
    cuDoubleComplex dataNew[3] = { 0 };
    for (int i=0; i<3; i++) {
        double s = 0, c = 0, m = 0;
        sincos(data[i].y, &s, &c);
        m = std::exp(data[i].x);
        dataNew[i] = make_cuDoubleComplex(m*c, m*s);
    }
    return ComplexVec(dataNew[0], dataNew[1], dataNew[2]);
}

Vec ComplexVec::realSign() {
    double dataNew[3] = { 0 };
    for (int i=0; i<3; i++) {
        dataNew[i] = data[i].x < 0 ? -1 : 1;
    }
    return Vec(dataNew[0], dataNew[1], dataNew[2]);
}

Vec ComplexVec::abs() {
    double dataNew[3] = { 0 };
    for (int i = 0; i<3; i++) {
        dataNew[i] = sqrt(data[i].x*data[i].x + data[i].y*data[i].y);
    }
    return Vec(dataNew[0], dataNew[1], dataNew[2]);
}

CUDA_CALLABLE_MEMBER cuDoubleComplex dot(const ComplexVec & a, const ComplexVec & b) {
    return (a * b).sum();
}

