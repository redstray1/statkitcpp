#pragma once

#include <immintrin.h>
#include "ScalarType.h"
#include "immintrin.h"
#include "UFunc.h"
#include "promote_type.h"

namespace statkitcpp {

namespace vec {

template <typename T, typename U>
void assign_scalar(void* data, const void* scalar, size_t size) { //NOLINT
    T* ptr_data = static_cast<T*>(data);
    const U* ptr_scalar = static_cast<const U*>(scalar);
    for (size_t i = 0; i < size; i++, ptr_data++) {
        *ptr_data = *ptr_scalar;
    } 
}

template <>
void assign_scalar<float, float>(void* data, const void* scalar, size_t size) {
    float* ptrdata = static_cast<float*>(data);
    const float* ptrscalar = static_cast<const float*>(scalar);
    __m256 zmm0;
    size_t i = 0;
    for (; i < (size&(~7)); i += 8, ptrdata += 8) {
        zmm0 = _mm256_broadcast_ss(ptrscalar);
        _mm256_storeu_ps(ptrdata, zmm0);
    }
    for (;i < size; i++, ptrdata++) {
        *ptrdata = *ptrscalar;
    }
}

template <>
void assign_scalar<double, double>(void* data, const void* scalar, size_t size) {
    double* ptrdata = static_cast<double*>(data);
    const double* ptrscalar = static_cast<const double*>(scalar);
    __m256d zmm0;
    size_t i = 0;
    for (; i < (size&(~3)); i += 4, ptrdata += 4) {
        zmm0 = _mm256_broadcast_sd(ptrscalar);
        _mm256_storeu_pd(ptrdata, zmm0);
    }
    for (;i < size; i++, ptrdata++) {
        *ptrdata = *ptrscalar;
    }
}

void assign_scalar(void* data, ScalarType dtype1, const void* scalar, ScalarType dtype2, size_t size) { //NOLINT
    DEFINE_FUNC_ARRAY_TEMPLATES(assign_scalar, void*, const void*, size_t);
    auto idx1 = kDType2Index[static_cast<int64_t>(dtype1)];
    auto idx2 = kDType2Index[static_cast<int64_t>(dtype2)];
    kFuncsLookup[idx1][idx2](data, scalar, size);
}

}

}