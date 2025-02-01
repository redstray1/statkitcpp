#pragma once

#include "../ScalarType.h"
#include "../Storage.h"
#include "../errors.h"
#include "config.h"
#include <vector>

namespace statkitcpp {

namespace ops {

template <typename T>
void min_deriv_(const T* data, //NOLINT
          size_t itemsize,
          size_t n_bytes,
          int dim,
          const std::vector<size_t>& shape,
          const std::vector<size_t>& strides,
          T* out,
          size_t out_bytes) {
    size_t max_index_left = (dim > 0 ? shape[0] * strides[0] / strides[dim - 1] : 1);
    size_t max_index_right = strides[dim];

    for (size_t left = 0; left < max_index_left; left++) {
        for (size_t right = 0; right < max_index_right; right++)  {
            size_t out_index = left * strides[dim] + right;
            bool started = false;
            T min_result = static_cast<T>(0);
            std::vector<size_t> min_positions;
            min_positions.reserve(shape[dim]);
            for (size_t k = 0; k < shape[dim]; k++) {
                size_t src_index = left * strides[dim] * shape[dim] + k * strides[dim] + right;
                if (out_index >= out_bytes / itemsize) {
                    throw OutOfRangeFlatError{out_index, out_bytes / itemsize};
                }
                if (src_index >= n_bytes / itemsize) {
                    throw OutOfRangeFlatError{src_index, n_bytes / itemsize};
                }
                out[src_index] = static_cast<T>(0);
                if (!started) {
                    min_result = data[src_index];
                    min_positions.push_back(src_index);
                    started = true;
                } else {
                    if (min_result > data[src_index]) {
                        min_result = data[src_index];
                        min_positions.clear();
                        min_positions.push_back(src_index);
                    } else if (abs(min_result - data[src_index]) < kMachinePrecision) {
                        min_positions.push_back(src_index);
                    }
                }
            }
            for (const auto& min_pos : min_positions) {
                out[min_pos] = static_cast<T>(1);
            }
        }
    }
}

void min_deriv(const Storage& data, //NOLINT
         ScalarType dtype,
         const std::vector<size_t>& shape,
         const std::vector<size_t>& strides,
         int dim,
         Storage& out) {
    #define DEFINE_TEMPLATE(T, name) \
    case (ScalarType::name): {\
        auto p = static_cast<const T*>(data.GetDataPtr()); \
        auto o = static_cast<T*>(out.GetDataPtr()); \
        min_deriv_<T>(p, ItemSize(dtype), data.GetNbytes(), dim, shape, strides, o, out.GetNbytes()); \
        break; \
    }
    switch (dtype) {
        SCALAR_TYPES(DEFINE_TEMPLATE)
        default:
            break;
    }
    #undef DEFINE_TEMPLATE
}
}
}