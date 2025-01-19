#pragma once

#include "../ScalarType.h"
#include "../Storage.h"
#include "../errors.h"
#include <vector>

namespace statkitcpp {

namespace ops {

template <typename T>
void mean_(const T* data, //NOLINT
          size_t itemsize,
          size_t n_bytes,
          int dim,
          const std::vector<size_t>& shape,
          const std::vector<size_t>& strides,
          T* out,
          size_t out_bytes) {
    size_t max_index_left = (dim > 0 ? shape[0] * strides[0] / strides[dim - 1] : 1);
    size_t max_index_right = strides[dim] / itemsize;

    for (size_t left = 0; left < max_index_left; left++) {
        for (size_t right = 0; right < max_index_right; right++)  {
            size_t out_index = left * strides[dim] / itemsize + right;
            out[out_index] = static_cast<T>(0);
            for (size_t k = 0; k < shape[dim]; k++) {
                size_t src_index = left * strides[dim] / itemsize * shape[dim] + k * strides[dim] / itemsize + right;
                if (out_index >= out_bytes / itemsize) {
                    throw OutOfRangeFlatError{out_index, out_bytes / itemsize};
                }
                if (src_index >= n_bytes / itemsize) {
                    throw OutOfRangeFlatError{src_index, n_bytes / itemsize};
                }
                out[out_index] += data[src_index];
            }
            out[out_index] /= shape[dim];
        }
    }
}

void mean(const Storage& data, //NOLINT
         ScalarType dtype,
         const std::vector<size_t>& shape,
         const std::vector<size_t>& strides,
         int dim,
         bool keepdims,
         Storage& out) {
    #define DEFINE_TEMPLATE(T, name) \
    case (ScalarType::name): {\
        auto p = static_cast<const T*>(data.GetDataPtr()); \
        auto o = static_cast<T*>(out.GetDataPtr()); \
        mean_<T>(p, ItemSize(dtype), data.GetNbytes(), dim, shape, strides, o, out.GetNbytes()); \
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