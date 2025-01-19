#include "tensor_creation_ops.h"
#include "ScalarType.h"
#include "errors.h"

namespace statkitcpp {

namespace ops {

void full(Tensor& data, const Scalar& fill_value) { //NOLINT
    #define DEFINE_TENSOR_FILL(T, name) \
        case(ScalarType::name): { \
            auto p = static_cast<T*>(data.GetDataPointer()); \
            std::fill(p, p + data.GetSize(), fill_value.to##name()); \
            break; \
        }   
    
    switch(data.GetDType()) {
        SCALAR_TYPES(DEFINE_TENSOR_FILL)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_TENSOR_FILL

}

void zeros(Tensor& data) { //NOLINT
    #define DEFINE_TENSOR_ZEROS(T, name) \
        case(ScalarType::name): { \
            auto p = static_cast<T*>(data.GetDataPointer()); \
            std::fill(p, p + data.GetSize(), static_cast<T>(0)); \
            break; \
        }   
    
    switch(data.GetDType()) {
        SCALAR_TYPES(DEFINE_TENSOR_ZEROS)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_TENSOR_ZEROS

}

void ones(Tensor& data) { //NOLINT
    #define DEFINE_TENSOR_ONES(T, name) \
        case(ScalarType::name): { \
            auto p = static_cast<T*>(data.GetDataPointer()); \
            std::fill(p, p + data.GetSize(), static_cast<T>(1)); \
            break; \
        }   
    
    switch(data.GetDType()) {
        SCALAR_TYPES(DEFINE_TENSOR_ONES)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_TENSOR_ONES
}

void arange(Tensor& data, const Scalar& start, const Scalar& end, const Scalar& step) {
    #define DEFINE_ARANGE_TYPE(T, name) \
    case (ScalarType::name):{ \
        T start_ = start.to##name(); \
        T end_ = end.to##name(); \
        T step_ = step.to##name(); \
        auto p = static_cast<T*>(data.GetDataPointer()); \
        for (size_t i = 0; i < data.GetSize(); i++) { \
            T value_ = start_ + static_cast<T>(i) * step_; \
            if (i >= data.GetSize() || i < 0) { \
                throw OutOfRangeFlatError{i, data.GetSize()}; \
            } \
            p[i] = value_; \
        } \
        break; \
    }
    switch(data.GetDType()) {
        SCALAR_TYPES(DEFINE_ARANGE_TYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_ARANGE_TYPE
}

}

}