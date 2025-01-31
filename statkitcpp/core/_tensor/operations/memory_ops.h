#include "../ScalarType.h"
#include "../Scalar.h"
#include "../shape.h"
#include "errors.h"
#include <vector>

namespace statkitcpp {

template <typename To>
To memconv(const void* data, ScalarType dtype) { //NOLINT
    #define DEFINE_DATA_TYPE(T, name) \
    case ScalarType::name: \
        return static_cast<To>(*(static_cast<const T*>(data)));
    switch(dtype) {
        SCALAR_TYPES(DEFINE_DATA_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_DATA_TYPE
}

template <typename Func>
void calc(const void* data1, ScalarType dtype1, //NOLINT
          const void* data2, ScalarType dtype2,
          void* out, ScalarType output_type, Func func) {
    #define DEFINE_OUTPUT_TYPE(T, name) \
    case ScalarType::name: {\
        T lhs = memconv<T>(data1, dtype1); \
        T rhs = memconv<T>(data2, dtype2); \
        *static_cast<T*>(out) = func(lhs, rhs); \
        break; \
    }
    switch(output_type) {
        SCALAR_TYPES(DEFINE_OUTPUT_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_OUTPUT_TYPE
}

namespace ops {

template <typename Out, typename T, typename Func>
void binary_op_(Out* data1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, size_t outsize, //NOLINT
                T* data2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
                Out* out, const std::vector<size_t>& out_strides, Func func, bool inverted=false) {
    for (size_t output_index = 0; output_index < outsize; output_index++) {
        size_t lhs_index = 0;
        size_t rhs_index = 0;

        size_t temp_output_idx = output_index;
        for (size_t dim = 0; dim < out_strides.size(); dim++) {
            int rev_dim = static_cast<int>(out_strides.size() - dim - 1);
            int lhs_dim = (static_cast<int>(shape1.size()) - rev_dim - 1);
            int rhs_dim = (static_cast<int>(shape2.size()) - rev_dim - 1);
            size_t out_idx = temp_output_idx / out_strides[dim];
            temp_output_idx %= out_strides[dim];

            if (lhs_dim >= 0 && lhs_dim < static_cast<int>(shape1.size())) {
                size_t lhs_dim_idx = 0;
                lhs_dim_idx = out_idx;
                if (shape1[lhs_dim] == 1) {
                    lhs_dim_idx = 0;
                }
                lhs_index += lhs_dim_idx * strides1[lhs_dim];
            }
            if (rhs_dim >= 0 && rhs_dim < static_cast<int>(shape2.size())) {
                size_t rhs_dim_idx = 0;
                rhs_dim_idx = out_idx;
                if (shape2[rhs_dim] == 1) {
                    rhs_dim_idx = 0;
                }
                rhs_index += rhs_dim_idx * strides2[rhs_dim];
            }
        }
        if (!inverted) {
            out[output_index] = func(data1[lhs_index], static_cast<Out>(data2[rhs_index]));
        } else {
            out[output_index] = func(static_cast<Out>(data2[rhs_index]), data1[lhs_index]);
        }
        
    }
}


template <typename Out, typename T, typename Func>
void scalar_op_(T* data1, size_t outsize, //NOLINT
                Scalar data2,
                Out* out, Func func) {
    for (size_t output_index = 0; output_index < outsize; output_index++) {
        out[output_index] = func(static_cast<T>(data1[output_index]), data2.to<Out>());
    }
}

template <typename Out, typename Func>
void temp_scalar_op(void* data1, ScalarType dtype1, size_t outsize, //NOLINT
                Scalar data2,
                Out* out, Func func) {
    #define DEFINE_TYPE(U, name) \
    case (ScalarType::name): \
        scalar_op_(static_cast<U*>(data1), outsize, \
                   data2, \
                   out, func); \
        break;
    switch(dtype1) {
        SCALAR_TYPES(DEFINE_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_TYPE
}


template <typename T, typename Func>
void temp_binary_op(T* data1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, size_t outsize, //NOLINT
                    void* data2, ScalarType dtype2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
                    T* out, const std::vector<size_t>& out_strides, Func func, bool inverted=false) {
    #define DEFINE_TYPE(U, name) \
    case (ScalarType::name): \
        binary_op_(data1, shape1, strides1, outsize, \
                   static_cast<U*>(data2), shape2, strides2, \
                   out, out_strides, func, inverted); \
        break;
    switch(dtype2) {
        SCALAR_TYPES(DEFINE_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_TYPE
}

template <typename Func>
void binary_op(void* data1, ScalarType dtype1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, size_t outsize, //NOLINT
               void* data2, ScalarType dtype2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
               void* out, ScalarType dtype, const std::vector<size_t>& out_strides, Func func) {
        #define DEFINE_DTYPE_LHS(T, name) \
        case (ScalarType::name): { \
            if (dtype1 == dtype) { \
                temp_binary_op(static_cast<T*>(data1), shape1, strides1, outsize, \
                               data2, dtype2, shape2, strides2, \
                               static_cast<T*>(out), out_strides, func); \
            } else if (dtype2 == dtype) { \
                temp_binary_op(static_cast<T*>(data2), shape2, strides2, outsize, \
                               data1, dtype1, shape1, strides1, \
                               static_cast<T*>(out), out_strides, func, true); \
            } \
            break; \
        }
        switch(dtype) {
            SCALAR_TYPES(DEFINE_DTYPE_LHS)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_DTYPE_LHS
}

template <typename Func>
void scalar_op(void* data1, ScalarType dtype1, size_t outsize, //NOLINT
               Scalar data2, void* out, ScalarType dtype, Func func) {
        #define DEFINE_DTYPE_LHS(T, name) \
        case (ScalarType::name): { \
            if (dtype1 == dtype) { \
                scalar_op_(static_cast<T*>(data1), outsize, data2, static_cast<T*>(out), func); \
            } else if (data2.Type() == dtype) { \
                temp_scalar_op(data1, dtype1, outsize, data2, static_cast<T*>(out), func); \
            } \
            break; \
        }
        switch(dtype) {
            SCALAR_TYPES(DEFINE_DTYPE_LHS)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_DTYPE_LHS
}

template <typename T, typename Func>
void pointwise_(T* data, size_t size, Func func, T* out) { //NOLINT
    for (size_t i = 0; i < size; i++) {
        out[i] = func(data[i]);
    }
}

template <typename Func>
void pointwise(void* data, size_t size, ScalarType type, Func func, void* out)  {//NOLINT
    #define DEFINE_DTYPE(T, name) \
    case (ScalarType::name): \
        pointwise_(static_cast<T*>(data), size, func, static_cast<T*>(out)); \
        break;
    switch(type) {
        SCALAR_TYPES_NOBOOL(DEFINE_DTYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_DTYPE
}
}

}