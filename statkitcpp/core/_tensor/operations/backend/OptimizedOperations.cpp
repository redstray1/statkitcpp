#include <functional>
#include "Operations.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "errors.h"
#include "shape.h"
#include "vectorized.h"
#include "binary_ops_memory.h"

namespace statkitcpp {

Tensor AddOptimized(const Tensor& lhs, const Tensor& rhs) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(output_shape, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        if (IsEqualSuffix(lhs.GetShape(), rhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                vec::shape2suff1_add(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), rhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else if (IsEqualSuffix(rhs.GetShape(), lhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                vec::shape2suff1_add(rhs.data_ptr<T>(), lhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), lhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else {
            ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::plus());
        }
    } else {
        ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::plus());
    }
    return output;
}

Tensor SubOptimized(const Tensor& lhs, const Tensor& rhs) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(output_shape, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        if (IsEqualSuffix(lhs.GetShape(), rhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                vec::shape2suff1_sub(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), rhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else if (IsEqualSuffix(rhs.GetShape(), lhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                vec::shape2suff1_sub(rhs.data_ptr<T>(), lhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), lhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else {
            ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::minus());
        }
    } else {
        ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::minus());
    }
    return output;
}

Tensor MulOptimized(const Tensor& lhs, const Tensor& rhs) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(output_shape, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        if (IsEqualSuffix(lhs.GetShape(), rhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name):\
                vec::shape2suff1_mul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), rhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else if (IsEqualSuffix(rhs.GetShape(), lhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                vec::shape2suff1_mul(rhs.data_ptr<T>(), lhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), lhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else {
            ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::multiplies());
        }
    } else {
        ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::multiplies());
    }
    return output;
}

Tensor DivOptimized(const Tensor& lhs, const Tensor& rhs) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(output_shape, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        if (IsEqualSuffix(lhs.GetShape(), rhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name):\
                vec::shape2suff1_div(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), rhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else if (IsEqualSuffix(rhs.GetShape(), lhs.GetShape())) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                vec::shape1suff2_div(rhs.data_ptr<T>(), lhs.data_ptr<T>(), output.data_ptr<T>(), \
                                 output.GetSize(), lhs.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else {
            ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::divides());
        }
    } else {
        ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(),
                   output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output_type, output.GetStrides(), std::divides());
    }
    return output;
}

}