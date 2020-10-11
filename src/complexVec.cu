//
//  complexVec.cpp
//  CUDA Physics
//
//  Created by Salvy Cavicchio on 10/11/20.
//  Copyright © 2020 Salvy Cavicchio. All rights reserved.
//

#include "vec.h"
#include <complex>
#include <thrust/complex.h>

thrust::complex<float> c = thrust::complex<float>(2.0f, 5.0f);
thrust::complex<float> c2 = c*c;
float r = c2.real();

