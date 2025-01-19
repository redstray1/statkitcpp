#include "../ScalarType.h"
#include "errors.h"

namespace statkitcpp {

template <typename To>
To memconv(const void* data, ScalarType dtype, size_t index) { //NOLINT
    #define DEFINE_DATA_TYPE(T, name) \
    case ScalarType::name: \
        return static_cast<To>(*(static_cast<const T*>(data) + index));
    switch(dtype) {
        SCALAR_TYPES(DEFINE_DATA_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_DATA_TYPE
}

template <typename Func>
void calc(const void* data1, ScalarType dtype1, size_t index1, //NOLINT
          const void* data2, ScalarType dtype2, size_t index2,
          void* out, ScalarType output_type, size_t out_index, Func func) {
    #define DEFINE_OUTPUT_TYPE(T, name) \
    case ScalarType::name: {\
        T lhs = memconv<T>(data1, dtype1, index1); \
        T rhs = memconv<T>(data2, dtype2, index2); \
        static_cast<T*>(out)[out_index] = func(lhs, rhs); \
        break; \
    }
    switch(output_type) {
        SCALAR_TYPES(DEFINE_OUTPUT_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
}

}