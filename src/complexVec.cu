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
	cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    data[0] = z;
    data[1] = z;
   	data[2] = z;
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

ComplexVec & ComplexVec::operator-=(const ComplexVec & v) {
	data[0] = cuCsub(data[0],v.data[0]);
    data[1] = cuCsub(data[1],v.data[1]);
    data[2] = cuCsub(data[2],v.data[2]);
    return *this;
}

ComplexVec & ComplexVec::operator*=(const ComplexVec & v) {
	data[0] = cuCmul(data[0],v.data[0]);
    data[1] = cuCmul(data[1],v.data[1]);
    data[2] = cuCmul(data[2],v.data[2]);
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
/* WORK IN PROGRESS
bool operator==(const ComplexVec & v1, const ComplexVec & v2) {
	bool real = (cuCreal(v1[0]) == cuCreal(v2[0]) && cuCreal(v1[1]) == cuCreal(v2[1]) && cuCreal(v1[2]) == cuCreal(v2[2]));
	bool imaj = (cuCimaj(v1[0]) == cuCimaj(v2[0]) && cuCimaj(v1[1]) == cuCimaj(v2[1]) && cuCimaj(v1[2]) == cuCimaj(v2[2]));
	return (real && imaj);
}


ComplexVec operator+(const ComplexVec & v1, const ComplexVec & v2) {
	return v1+=v2;
}

ComplexVec operator-(const ComplexVec & v1, const ComplexVec & v2) {
	return v1-=v2;
}

ComplexVec operator*(const ComplexVec & v1, const ComplexVec & v2) {
	return v1*=v2;
}

ComplexVec operator/(const ComplexVec & v1, const ComplexVec & v2) {
	return v1/=v2;
}

std::ostream & operator << (std::ostream & strm, const ComplexVec & v) {
	return strm << "(" << cuCreal(v[0]) << " + " << cuCimag(v[0]) << "i" << ", " << cuCreal(v[1]) << " + " << cuCimaj(v[0]) << "i" << ", " << cuCreal(v[2]) << " + " << cuCimag(v[0]) << "i)";
}


void ComplexVec::print() {
	printf("(%3f + %3fi, %3f + %3fi, %3f + %3fi)\n", cuCreal(data[0]),cuCimaj(data[0]), cuCreal(data[1]),cuCimaj(data[1]), cuCreal(data[2]),cuCimaj(data[2]));
}

cuDoubleComplex ComplexVec::sum() const {
	return make_cuDoubleComplex(cuCadd(cuCadd(data[0],data[1],data[2])));
}
*/
