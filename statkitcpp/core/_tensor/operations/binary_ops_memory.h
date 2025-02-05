#pragma once

#include "../_utils/UFunc.h"
#include <cstdint>
#include <cstddef>
#include "ScalarType.h"
#include "promote_type.h"
#include "TensorIterator.h"
#include "Array.h"
#include <chrono>
#include <iostream>

namespace statkitcpp {
namespace ops {

#define DEFINE_BINARY2(type1, type2, out_type) \
template <typename Func> \
void type1##_##type2##_binary_op(TensorIteratorPair& iter, /*NOLINT*/  \
                void* out, const Func& func) { \
    auto start = std::chrono::high_resolution_clock::now(); \
    for (size_t output_index = 0; !iter.Expired(); output_index++) { \
\
        static_cast<out_type*>(out)[output_index] \
            = func(static_cast<out_type>(*static_cast<type1*>(iter.IterF())), \
                   static_cast<out_type>(*static_cast<type2*>(iter.IterS()))); \
        ++iter; \
    } \
    auto stop = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); \
    std::cout << "Time taken by inner function binary_op: " << duration.count() << " ms" << '\n'; \
}

#define DEFINE_BINARY(type1, type2, out_type) \
template <typename Func> \
void type1##_##type2##_binary_op(void* data1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, const size_t& outsize, /*NOLINT*/ \
                 void* data2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2, \
                 void* out, const std::vector<size_t>& out_strides, const Func& func) { \
    type1* data1_t = static_cast<type1*>(data1); \
    type2* data2_t = static_cast<type2*>(data2); \
    out_type* out_t = static_cast<out_type*>(out); \
    auto lhs_ptr = data1_t; \
    auto rhs_ptr = data2_t; \
    for (size_t output_index = 0; output_index < outsize; ++output_index, ++out_t) { \
        lhs_ptr = data1_t; \
        rhs_ptr = data2_t; \
        size_t temp_output_idx = output_index; \
        int lhs_dim = (static_cast<int>(strides1.size()) - out_strides.size()); \
        int rhs_dim = (static_cast<int>(strides2.size()) - out_strides.size()); \
        int dim = 0; \
        if (lhs_dim < 0) { \
            for (; dim < static_cast<int>(out_strides.size() - strides1.size()); ++dim, ++lhs_dim, ++rhs_dim) { \
                size_t out_idx = temp_output_idx / out_strides[dim]; \
                temp_output_idx -= out_idx * out_strides[dim]; \
                size_t rhs_dim_idx = out_idx; \
                if (shape2[rhs_dim] == 1) { \
                    rhs_dim_idx = 0; \
                } \
                rhs_ptr +=rhs_dim_idx * strides2[rhs_dim]; \
            } \
        } else { \
            for (; dim < static_cast<int>(out_strides.size() - strides2.size()); ++dim, ++lhs_dim, ++rhs_dim) { \
                size_t out_idx = temp_output_idx / out_strides[dim]; \
                temp_output_idx -= out_idx * out_strides[dim]; \
                size_t lhs_dim_idx = out_idx; \
                if (shape1[lhs_dim] == 1) { \
                    lhs_dim_idx = 0; \
                } \
                lhs_ptr += lhs_dim_idx * strides1[lhs_dim]; \
            } \
        } \
        for (; dim < static_cast<int>(out_strides.size()); ++dim, ++lhs_dim, ++rhs_dim) { \
            size_t out_idx = temp_output_idx / out_strides[dim]; \
            temp_output_idx -= out_idx * out_strides[dim]; \
            size_t dim_idx = out_idx; \
            if (shape2[rhs_dim] == 1) { \
                dim_idx = 0; \
            } \
            rhs_ptr += dim_idx * strides2[rhs_dim]; \
            dim_idx = out_idx; \
            if (shape1[lhs_dim] == 1) { \
                dim_idx = 0; \
            } \
            lhs_ptr += dim_idx * strides1[lhs_dim]; \
        } \
        *out_t \
            = func(static_cast<out_type>(*lhs_ptr), \
                   static_cast<out_type>(*rhs_ptr)); \
    } \
}

#define DEFINE_BINARY3(type1, type2, out_type) \
template <typename Func> \
void type1##_##type2##_binary_op(void*& data1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, const size_t& outsize, /*NOLINT*/ \
                 void*& data2, [[maybe_unused]]const std::vector<size_t>& shape2, [[maybe_unused]]const std::vector<size_t>& strides2, \
                 void*& out, [[maybe_unused]]const std::vector<size_t>& out_strides, [[maybe_unused]]const Func& func) { \
    type1* data1_t = static_cast<type1*>(data1); \
    type2* data2_t = static_cast<type2*>(data2); \
    out_type* out_t = static_cast<out_type*>(out); \
    auto lhs_ptr = data1_t; \
    auto rhs_ptr = data2_t; \
    for (size_t output_index = 0; output_index < outsize; ++output_index, ++out_t, ++lhs_ptr, ++rhs_ptr) { \
        *out_t \
            = func(*lhs_ptr, \
                   *rhs_ptr); \
    } \
}

// template <typename Tp1, typename Tp2, typename Out, typename Func> \
// void binary_op_(void* data1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, size_t outsize, /*NOLINT*/
//                 void* data2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
//                 void* out, const std::vector<size_t>& out_strides, const Func& func) { 
//     for (size_t output_index = 0; output_index < outsize; output_index++) {
//         size_t lhs_index = 0;
//         size_t rhs_index = 0;
//         size_t temp_output_idx = output_index;
//         for (size_t dim = 0; dim < out_strides.size(); dim++) {
//             int rev_dim = static_cast<int>(out_strides.size() - dim - 1);
//             int lhs_dim = (static_cast<int>(shape1.size()) - rev_dim - 1);
//             int rhs_dim = (static_cast<int>(shape2.size()) - rev_dim - 1);
//             size_t out_idx = temp_output_idx / out_strides[dim];
//             temp_output_idx %= out_strides[dim];
//             if (lhs_dim >= 0 && lhs_dim < static_cast<int>(shape1.size())) {
//                 size_t lhs_dim_idx = 0;
//                 lhs_dim_idx = out_idx;
//                 if (shape1[lhs_dim] == 1) {
//                     lhs_dim_idx = 0;
//                 }
//                 lhs_index += lhs_dim_idx * strides1[lhs_dim];
//             }
//             if (rhs_dim >= 0 && rhs_dim < static_cast<int>(shape2.size())) {
//                 size_t rhs_dim_idx = 0;
//                 rhs_dim_idx = out_idx;
//                 if (shape2[rhs_dim] == 1) {
//                     rhs_dim_idx = 0;
//                 }
//                 rhs_index += rhs_dim_idx * strides2[rhs_dim];
//             }
//         }
//         static_cast<Out*>(out)[output_index]
//             = func(static_cast<Out>(static_cast<Tp1*>(data1)[lhs_index]),
//                    static_cast<Out>(static_cast<Tp2*>(data2)[rhs_index]));
//     }
// }

DEFINE_TYPES_OF_BINARY_ARGS(DEFINE_BINARY)

template <typename Func>
void binary_op(void* data1, ScalarType dtype1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, const size_t& outsize, //NOLINT
               void* data2, ScalarType dtype2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
               void* out, [[maybe_unused]]ScalarType dtype, const std::vector<size_t>& out_strides, const Func& func) {
    DEFINE_FUNC_ARRAY(binary_op, binary_op<Func>, void*, const std::vector<size_t>&, const std::vector<size_t>&, const size_t&,
                      void*, const std::vector<size_t>&, const std::vector<size_t>&, void*, const std::vector<size_t>&, const Func&)
    auto idx1 = kDType2Index[static_cast<int64_t>(dtype1)];
    auto idx2 = kDType2Index[static_cast<int64_t>(dtype2)];
    kFuncsLookup[idx1][idx2](data1, shape1, strides1, outsize,
                               data2, shape2, strides2,
                               out, out_strides, func);
}

}
}