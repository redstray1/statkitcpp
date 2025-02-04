#include <immintrin.h>
#include <iostream>

namespace statkitcpp {

namespace vec {

template <typename T>
void dot(T* data1, T* data2, T* out, const size_t& size) { //NOLINT
    *out = 0;
    for (size_t i = 0; i < size; ++i, ++data1,++data2) {
        *out += *data1 * (*data2);
    }
}


template <>
void dot<int>(int* data1, int* data2, int* out, const size_t& size) {
    __m512i zmm0, zmm1;
    size_t i =0;
    *out = 0;
    for (; i < (size&(~15)); i += 16, data1 += 16, data2 += 16) {
        zmm0 = _mm512_loadu_epi32(data1);
        zmm1 = _mm512_loadu_epi32(data2);
        zmm0 = _mm512_mul_epi32(zmm0, zmm1);
        *out += _mm512_reduce_add_epi32(zmm0);
    }
    for (; i < size; ++i, ++data1, ++data2) {
        *out += *data1 * (*data2);
    }
}

template <>
void dot<int64_t>(int64_t* data1, int64_t* data2, int64_t* out, const size_t& size) {
    __m512i zmm0, zmm1;
    size_t i =0;
    *out = 0;
    for (; i < (size&(~7)); i += 8, data1 += 8, data2 += 8) {
        zmm0 = _mm512_loadu_epi64(data1);
        zmm1 = _mm512_loadu_epi64(data2);
        zmm0 = _mm512_mullox_epi64(zmm0, zmm1);
        *out += _mm512_reduce_add_epi64(zmm0);
    }
    for (; i < size; ++i, ++data1, ++data2) {
        *out += *data1 * (*data2);
    }
}

template <>
void dot<float>(float* data1, float* data2, float* out, const size_t& size) {
    __m512 zmm0, zmm1, zmm2;
    zmm2 = _mm512_setzero_ps();
    size_t i =0;
    *out = 0;
    for (; i < (size&(~15)); i += 16, data1 += 16, data2 += 16) {
        zmm0 = _mm512_loadu_ps(data1);
        zmm1 = _mm512_loadu_ps(data2);
        zmm2 = _mm512_fmadd_ps(zmm0, zmm1, zmm2);
    }
    *out = _mm512_reduce_add_ps(zmm2);
    for (; i < size; ++i, ++data1, ++data2) {
        *out += *data1 * (*data2);
    }
}

template <>
void dot<double>(double* data1, double* data2, double* out, const size_t& size) {
    __m512d zmm0, zmm1, zmm2;
    zmm2 = _mm512_setzero_pd();
    size_t i =0;
    *out = 0;
    for (; i < (size&(~7)); i += 8, data1 += 8, data2 += 8) {
        zmm0 = _mm512_loadu_pd(data1);
        zmm1 = _mm512_loadu_pd(data2);
        zmm2 = _mm512_fmadd_pd(zmm0, zmm1, zmm2);
    }
    *out = _mm512_reduce_add_pd(zmm2);
    for (; i < size; ++i, ++data1, ++data2) {
        *out += *data1 * (*data2);
    }
}

template <typename T>
void matmatmul_transposed(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& k) { //NOLINT
    for (size_t idx = 0; idx < n * k; ++idx, ++out) {
        dot(mat1 + (idx / k) * m, mat2 + (idx % k) * m, out, m);
    }
}

template <typename T>
void matmatmul(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& k) { //NOLINT
    for (size_t idx = 0; idx < n * k; ++idx, ++out) {
        *out = 0;
        for (size_t p = 0; p < m; p++) {
            *out += 
                *(mat1 + (idx / k) * m + p) * 
                *(mat2 + p * k + idx % k);
        }
    }
}


template <typename T>
void vecmatmul_transposed(T* mat1, T* mat2, T* out, const size_t& m, const size_t& k) { //NOLINT
    for (size_t idx = 0; idx < k; ++idx, ++out) {
        dot(mat1, mat2 + (idx % k) * m, out, m);
    }
}

template <typename T>
void vecmatmul(T* mat1, T* mat2, T* out, const size_t& m, const size_t& k) { //NOLINT
    for (size_t idx = 0; idx < k; ++idx, ++out) {
        *out = 0;
        for (size_t p = 0; p < m; p++) {
            *out += *(mat1 + p) * *(mat2 + p * k + idx % k);
        }
    }
}

template <typename T>
void matvecmul(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m) { //NOLINT
    for (size_t idx = 0; idx < n; ++idx, ++out) {
        dot(mat1 + idx * m, mat2, out, m);
    }
}

template <typename T>
void batched_vecmatmul(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += m, mat2 += n * m, out += m) {
        vecmatmul(mat1, mat2, out, n, m);
    }
}

template <typename T>
void batched_vecmatmul_transposed(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += m, mat2 += n * m, out += m) {
        vecmatmul_transposed(mat1, mat2, out, n, m);
    }
}

template <typename T>
void batched_matvecmul(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += n, mat1 += n * m, out += n) {
        matvecmul(mat1, mat2, out, n, m);
    }
}

template <typename T>
void batched_mat2matmul(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& k, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += n * k, mat2 += m * k, out += n * k) {
        matmatmul(mat1, mat2, out, n, m, k);
    }
}

template <typename T>
void batched_mat2matmul_transposed(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& k, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += n * k, mat2 += m * k, out += n * k) {
        matmatmul_transposed(mat1, mat2, out, n, m, k);
    }
}

template <typename T>
void batched_matmat2mul(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& k, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += n * k, mat1 += n * m, out += n * k) {
        matmatmul(mat1, mat2, out, n, m, k);
    }
}

template <typename T>
void batched_matmat2mul_transposed(T* mat1, T* mat2, T* out, const size_t& n, const size_t& m, const size_t& k, const size_t& n_size) { //NOLINT
    for (size_t idx = 0; idx < n_size; idx += n * k, mat1 += n * m, out += n * k) {
        matmatmul_transposed(mat1, mat2, out, n, m, k);
    }
}

}

}