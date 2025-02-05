#pragma once

#include <immintrin.h>
#include <xmmintrin.h>
#include <cstddef>
#include "errors.h"

namespace statkitcpp {

namespace vec {


template <typename T>
void transpose_4pack(T* src, T* dest, const size_t& n, const size_t& m) { //NOLINT
    size_t const max_i = (n / 4) * 4;
    size_t const max_j = (m / 4) * 4;

    __m128 r0, r1, r2, r3;

    for (size_t i = 0; i != max_i; i += 4) {
        for (size_t j = 0; j != max_j; j += 4) {
            r0 = _mm_loadu_ps(reinterpret_cast<float*>(src + i * m + j));
            r1 = _mm_loadu_ps(reinterpret_cast<float*>(src + (i + 1) * m + j));
            r2 = _mm_loadu_ps(reinterpret_cast<float*>(src + (i + 2) * m + j));
            r3 = _mm_loadu_ps(reinterpret_cast<float*>(src + (i + 3) * m + j));

            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

            _mm_storeu_ps(reinterpret_cast<float*>(dest + j * n + i),r0);
            _mm_storeu_ps(reinterpret_cast<float*>(dest + (j + 1) * n + i),r1);
            _mm_storeu_ps(reinterpret_cast<float*>(dest + (j + 2) * n + i),r2);
            _mm_storeu_ps(reinterpret_cast<float*>(dest + (j + 3) * n + i),r3);
        }
        for (size_t k = i; k != n; ++k) {
            for (size_t j = max_j; j != m; ++j) {
                dest[j * n + i] = src[k * m + j];
            }
        }
    }
    for (size_t i = max_i; i != n; ++i) {
        for (size_t j = 0; j != m; ++j) {
            dest[j * n + i] = src[i * m + j];
        }
    }
}

template <typename T>
void transpose_2pack(T* src, T* dest, const size_t& n, const size_t& m) { //NOLINT
    size_t const max_i = (n / 2) * 2;
    size_t const max_j = (m / 2) * 2;

    __m128d r0, r1;

    if (max_j != m) {
        for (size_t i = 0; i < max_i; i += 2) {
            for (size_t j = 0; j < max_j; j += 2) {
                r0 = _mm_loadu_pd(reinterpret_cast<double*>(src + i * m + j));
                r1 = _mm_loadu_pd(reinterpret_cast<double*>(src + (i + 1) * m + j));

                _mm_storeu_pd(reinterpret_cast<double*>(dest + j * n + i), _mm_shuffle_pd(r0, r1, 0b00));
                _mm_storeu_pd(reinterpret_cast<double*>(dest + (j + 1) * n + i), _mm_shuffle_pd(r0, r1, 0b11));
            }
            for (size_t k = i; k != n; ++k) {
                for (size_t j = max_j; j != m; ++j) {
                    dest[j * n + i] = src[k * m + j];
                }
            }
        }
        for (size_t i = max_i; i != n; ++i) {
            for (size_t j = 0; j != m; ++j) {
                dest[j * n + i] = src[i * m + j];
            }
        }
    } else {
        for (size_t i = 0; i < max_i; i += 2) {
            for (size_t j = 0; j < max_j; j += 2) {
                r0 = _mm_loadu_pd(reinterpret_cast<double*>(src + i * m + j));
                r1 = _mm_loadu_pd(reinterpret_cast<double*>(src + (i + 1) * m + j));

                _mm_storeu_pd(reinterpret_cast<double*>(dest + j * n + i), _mm_shuffle_pd(r0, r1, 0b00));
                _mm_storeu_pd(reinterpret_cast<double*>(dest + (j + 1) * n + i), _mm_shuffle_pd(r0, r1, 0b11));
            }
            for (size_t k = i; k != n; ++k) {
                for (size_t j = max_j; j != m; ++j) {
                    dest[j * n + i] = src[k * m + j];
                }
            }
        }
        if (max_i != n) {
            for (size_t j = 0; j < m; j++) {
                dest[j * n + max_i] = src[max_i * m + j];
            }
         }
    }
}

template <typename T>
void transpose(T* src, T* dest, const size_t& n, const size_t& m) { //NOLINT
    constexpr size_t kDataSize = sizeof(T);
    if constexpr (4 == kDataSize) {
        transpose_4pack(src, dest, n, m);
    } else if constexpr (8 == kDataSize) {
        transpose_2pack(src, dest, n, m);
    } else {
        throw NotImplemetedError{"Transpose not implemeted for types other than float and double"};
    }
}

template <typename T>
void batched_transpose(T* src, T* dest, const size_t& n, const size_t& m, const size_t& outsize) { //NOLINT
    for (size_t i = 0; i < outsize; i += n * m, src += n * m, dest += n * m) {
        transpose(src, dest, n, m);
    }
}

}

}