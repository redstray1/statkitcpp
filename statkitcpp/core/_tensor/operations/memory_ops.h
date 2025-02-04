#include "../ScalarType.h"
#include "../Scalar.h"
#include "../shape.h"
#include "errors.h"
#include <vector>

namespace statkitcpp {

namespace ops {

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

// template <typename Func>
// void binary_op(void* data1, ScalarType dtype1, const std::vector<size_t>& shape1, const std::vector<size_t>& strides1, size_t outsize, //NOLINT
//                void* data2, ScalarType dtype2, const std::vector<size_t>& shape2, const std::vector<size_t>& strides2,
//                void* out, ScalarType dtype, const std::vector<size_t>& out_strides, Func func);

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