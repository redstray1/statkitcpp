#pragma once
#include <immintrin.h>
#include "Tensor.h"
#include "immintrin.h"
#include <cstdint>
#include <iostream>

namespace statkitcpp {

namespace vec {

template <typename T>
void same_shape_add(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize) {
    for (size_t out_idx = 0; out_idx < outsize; ++out_idx, ++out, ++data1, ++data2) {
        *out = *data1 + *data2;
    }
}

template <>
void same_shape_add<int>(int* data1, int* data2, int* out, const size_t& n_bytes) { //NOLINT
    __m512i zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_epi32(data1);
        zmm1 = _mm512_loadu_epi32(data2);
        zmm0 = _mm512_add_epi32(zmm0, zmm1);
        _mm512_storeu_epi32(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 + *data2;
    }
}

template <>
void same_shape_add<int64_t>(int64_t* data1, int64_t* data2, int64_t* out, const size_t& n_bytes) { //NOLINT
    __m512i zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_epi64(data1);
        zmm1 = _mm512_loadu_epi64(data2);
        zmm0 = _mm512_add_epi64(zmm0, zmm1);
        _mm512_storeu_epi64(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 + *data2;
    }
}

template <>
void same_shape_add<float>(float* data1, float* data2, float* out, const size_t& n_bytes) { //NOLINT
    __m512 zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_ps(data1);
        zmm1 = _mm512_loadu_ps(data2);
        zmm0 = _mm512_add_ps(zmm0, zmm1);
        _mm512_storeu_ps(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 + *data2;
    }
}

template <>
void same_shape_add<double>(double* data1, double* data2, double* out, const size_t& n_bytes) { //NOLINT
    __m512d zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_pd(data1);
        zmm1 = _mm512_loadu_pd(data2);
        zmm0 = _mm512_add_pd(zmm0, zmm1);
        _mm512_storeu_pd(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 + *data2;
    }
}

template <typename T>
void shape2suff1_add(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize, const size_t& stride) {
    for (size_t out_idx = 0; out_idx < outsize; out_idx += stride, out += stride, data1 += stride) {
        same_shape_add(data1, data2, out, stride);
    }
}

template <typename T>
void same_shape_sub(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize) {
    for (size_t out_idx = 0; out_idx < outsize; ++out_idx, ++out, ++data1, ++data2) {
        *out = *data1 - *data2;
    }
}

template <>
void same_shape_sub<int>(int* data1, int* data2, int* out, const size_t& n_bytes) { //NOLINT
    __m512i zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_epi32(data1);
        zmm1 = _mm512_loadu_epi32(data2);
        zmm0 = _mm512_sub_epi32(zmm0, zmm1);
        _mm512_storeu_epi32(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 - *data2;
    }
}

template <>
void same_shape_sub<int64_t>(int64_t* data1, int64_t* data2, int64_t* out, const size_t& n_bytes) { //NOLINT
    __m512i zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_epi64(data1);
        zmm1 = _mm512_loadu_epi64(data2);
        zmm0 = _mm512_sub_epi64(zmm0, zmm1);
        _mm512_storeu_epi64(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 - *data2;
    }
}

template <>
void same_shape_sub<float>(float* data1, float* data2, float* out, const size_t& n_bytes) { //NOLINT
    __m512 zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_ps(data1);
        zmm1 = _mm512_loadu_ps(data2);
        zmm0 = _mm512_sub_ps(zmm0, zmm1);
        _mm512_storeu_ps(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 - *data2;
    }
}

template <>
void same_shape_sub<double>(double* data1, double* data2, double* out, const size_t& n_bytes) { //NOLINT
    __m512d zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_pd(data1);
        zmm1 = _mm512_loadu_pd(data2);
        zmm0 = _mm512_sub_pd(zmm0, zmm1);
        _mm512_storeu_pd(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 - *data2;
    }
}

template <typename T>
void shape2suff1_sub(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize, const size_t& stride) {
    for (size_t out_idx = 0; out_idx < outsize; out_idx += stride, out += stride, data1 += stride) {
        same_shape_sub(data1, data2, out, stride);
    }
}

template <typename T>
void same_shape_mul(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize) {
    for (size_t out_idx = 0; out_idx < outsize; ++out_idx, ++out, ++data1, ++data2) {
        *out = (*data1) * (*data2);
    }
}


template <>
void same_shape_mul<int>(int* data1, int* data2, int* out, const size_t& n_bytes) { //NOLINT
    __m512i zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_epi32(data1);
        zmm1 = _mm512_loadu_epi32(data2);
        zmm0 = _mm512_mul_epi32(zmm0, zmm1);
        _mm512_storeu_epi32(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 * (*data2);
    }
}

template <>
void same_shape_mul<int64_t>(int64_t* data1, int64_t* data2, int64_t* out, const size_t& n_bytes) { //NOLINT
    __m512i zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_epi64(data1);
        zmm1 = _mm512_loadu_epi64(data2);
        zmm0 = _mm512_mullox_epi64(zmm0, zmm1);
        _mm512_storeu_epi64(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 * (*data2);
    }
}

template <>
void same_shape_mul<float>(float* data1, float* data2, float* out, const size_t& n_bytes) { //NOLINT
    __m512 zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_ps(data1);
        zmm1 = _mm512_loadu_ps(data2);
        zmm0 = _mm512_mul_ps(zmm0, zmm1);
        _mm512_storeu_ps(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 * (*data2);
    }
}

template <>
void same_shape_mul<double>(double* data1, double* data2, double* out, const size_t& n_bytes) { //NOLINT
    __m512d zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_pd(data1);
        zmm1 = _mm512_loadu_pd(data2);
        zmm0 = _mm512_mul_pd(zmm0, zmm1);
        _mm512_storeu_pd(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 * (*data2);
    }
}

template <typename T>
void shape2suff1_mul(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize, const size_t& stride) {
    for (size_t out_idx = 0; out_idx < outsize; out_idx += stride, out += stride, data1 += stride) {
        same_shape_mul(data1, data2, out, stride);
    }
}



//Multiplication with intrinsics

template <typename T>
void same_shape_div(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize) {
    for (size_t out_idx = 0; out_idx < outsize; ++out_idx, ++out, ++data1, ++data2) {
        *out = (*data1) / (*data2);
    }
}

template <>
void same_shape_div<float>(float* data1, float* data2, float* out, const size_t& n_bytes) { //NOLINT
    __m512 zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~15)); i += 16, data1 += 16, data2 += 16, out += 16) {
        zmm0 = _mm512_loadu_ps(data1);
        zmm1 = _mm512_loadu_ps(data2);
        zmm0 = _mm512_div_ps(zmm0, zmm1);
        _mm512_storeu_ps(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 / (*data2);
    }
}

template <>
void same_shape_div<double>(double* data1, double* data2, double* out, const size_t& n_bytes) { //NOLINT
    __m512d zmm0, zmm1;
    size_t i =0;
    for (; i < (n_bytes&(~7)); i += 8, data1 += 8, data2 += 8, out += 8) {
        zmm0 = _mm512_loadu_pd(data1);
        zmm1 = _mm512_loadu_pd(data2);
        zmm0 = _mm512_div_pd(zmm0, zmm1);
        _mm512_storeu_pd(out, zmm0);
    }
    for (; i < n_bytes; i++, ++data1, ++data2, ++out) {
        *out = *data1 / (*data2);
    }
}

template <typename T>
void shape2suff1_div(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize, const size_t& stride) {
    for (size_t out_idx = 0; out_idx < outsize; out_idx += stride, out += stride, data1 += stride) {
        same_shape_div(data1, data2, out, stride);
    }
}

template <typename T>
void shape1suff2_div(T* data1, //NOLINT
               T* data2,
               T* out, const size_t& outsize, const size_t& stride) {
    for (size_t out_idx = 0; out_idx < outsize; out_idx += stride, out += stride, data2 += stride) {
        same_shape_div(data1, data2, out, stride);
    }
}

void optimized_add(ScalarType dtype, void* data1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, //NOLINT
               void* data2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
               void* out, const std::vector<size_t>& out_strides, const size_t& outsize);

}

}