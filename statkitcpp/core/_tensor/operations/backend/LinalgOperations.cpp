#include <stdexcept>
#include "Operations.h"
#include "linalg_vectorized.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "errors.h"
#include "shape.h"

namespace statkitcpp {

Tensor MatMulImpl(const Tensor& lhs, const Tensor& rhs, bool rhs_transposed) {
    if (lhs.GetNDim() == 1 && rhs.GetNDim() == 1) {
        return DotImpl(lhs, rhs);
    }
    if (lhs.GetNDim() == 2 && rhs.GetNDim() == 2) {
        return MatMatMulImpl(lhs, rhs, rhs_transposed);
    }
    if (lhs.GetNDim() == 1 && rhs.GetNDim() == 2) {
        return VecMatImpl(lhs, rhs);
    }
    if (lhs.GetNDim() == 2 && rhs.GetNDim() == 1) {
        return MatVecImpl(lhs, rhs);
    }
    if (lhs.GetNDim() > 2 || rhs.GetNDim() > 2) {
        return BatchedMatMatMulImpl(lhs, rhs);
    }
    throw std::runtime_error{"Matrix multiplication error"};
}

Tensor DotImpl(const Tensor &lhs, const Tensor &rhs) {
    if (lhs.GetSize() != rhs.GetSize()) {
        throw DotOperationError{lhs.GetSize(), rhs.GetSize()};
    }
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output({1}, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            vec::dot(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetSize()); \
            break;
        switch(output_type) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
    } else {
        throw NotImplemetedError{"Dot product for different types not implemented yet"};
    }
    return output;
}

Tensor MatMatMulImpl(const Tensor &lhs, const Tensor &rhs, bool rhs_transposed) {
    if (lhs.GetDim(1) != rhs.GetDim(rhs_transposed)) {
        throw MatMulError{lhs.GetDim(1), rhs.GetDim(rhs_transposed)};
    }
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output({lhs.GetDim(0), rhs.GetDim(1 - rhs_transposed)}, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            if (rhs_transposed) { \
                vec::matmatmul_transposed(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), lhs.GetDim(1), output.GetDim(1)); \
            } else {\
                vec::matmatmul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), lhs.GetDim(1), output.GetDim(1)); \
            } \
            break;
        switch(output_type) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
    } else {
        throw NotImplemetedError{"Matrix multiplication for different types not implemented yet"};
    }
    return output;
}

Tensor VecMatImpl(const Tensor &lhs, const Tensor &rhs, bool rhs_transposed) {
    if (lhs.GetDim(0) != rhs.GetDim(rhs_transposed)) {
        throw MatMulError{lhs.GetDim(0), rhs.GetDim(rhs_transposed)};
    }
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output({rhs.GetDim(1 - rhs_transposed)}, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            if (rhs_transposed) { \
                vec::vecmatmul_transposed(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), output.GetDim(1)); \
            } else {\
                vec::vecmatmul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), output.GetDim(1)); \
            } \
            break;
        switch(output_type) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
    } else {
        throw NotImplemetedError{"Vector-Matrix multiplication for different types not implemented yet"};
    }
    return output;
}

Tensor MatVecImpl(const Tensor &lhs, const Tensor &rhs) {
    if (lhs.GetDim(1) != rhs.GetDim(0)) {
        throw MatMulError{lhs.GetDim(1), rhs.GetDim(0)};
    }
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output({lhs.GetDim(0)}, output_type);
    if (lhs.GetDType() == rhs.GetDType()) {
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            vec::matvecmul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), lhs.GetDim(1)); \
            break;
        switch(output_type) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
    } else {
        throw NotImplemetedError{"Matrix-vector multiplication for different types not implemented yet"};
    }
    return output;
}

Tensor BatchedMatMatMulImpl(const Tensor &lhs, const Tensor &rhs, bool rhs_transposed) { //NOLINT
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(BroadcastShapesForMatMul(lhs.GetShape(), rhs.GetShape(), rhs_transposed), output_type);
    if (lhs.GetNDim() == 1) {
        if (lhs.GetDim(0) != rhs.GetDim(-2 + rhs_transposed)) {
            throw MatMulError{lhs.GetDim(0), rhs.GetDim(-2 + rhs_transposed)};
        }
        if (lhs.GetDType() == rhs.GetDType()) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                if (rhs_transposed) { \
                    vec::batched_vecmatmul_transposed(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), rhs.GetDim(-1), output.GetSize()); \
                } else {\
                    vec::batched_vecmatmul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(0), rhs.GetDim(-2), output.GetSize()); \
                } \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else {
            throw NotImplemetedError{"Batched Vector-Matrix multiplication for different types not implemented yet"};
        }
        return output;
    }
    if (rhs.GetNDim() == 1) {
        if (lhs.GetDim(-1) != rhs.GetDim(0)) {
            throw MatMulError{lhs.GetDim(-1), rhs.GetDim(0)};
        }
        if (lhs.GetDType() == rhs.GetDType()) {
            #define DEFINE_TYPE(T, name) \
            case (ScalarType::name): \
                    vec::batched_matvecmul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(-2), lhs.GetDim(-1), output.GetSize()); \
                break;
            switch(output_type) {
                SCALAR_TYPES(DEFINE_TYPE)
                default:
                    throw InvalidDatatypeError{};
            }
            #undef DEFINE_TYPE
        } else {
            throw NotImplemetedError{"Batched Matrix-Vector multiplication for different types not implemented yet"};
        }
        return output;
    }
    if (lhs.GetNDim() == 2) {
        if (lhs.GetDim(-1) != rhs.GetDim(-2 + rhs_transposed)) {
            throw MatMulError{lhs.GetDim(-1), rhs.GetDim(-2 + rhs_transposed)};
        }
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            if (rhs_transposed) { \
                vec::batched_mat2matmul_transposed(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(-2), lhs.GetDim(-1), rhs.GetDim(-2), output.GetSize()); \
            } else {\
                vec::batched_mat2matmul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(-2), lhs.GetDim(-1), rhs.GetDim(-1), output.GetSize()); \
            } \
            break;
        switch(output_type) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
        return output;
    }
    if (rhs.GetNDim() == 2) {
        if (lhs.GetDim(-1) != rhs.GetDim(-2 + rhs_transposed)) {
            throw MatMulError{lhs.GetDim(-1), rhs.GetDim(-2 + rhs_transposed)};
        }
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            if (rhs_transposed) { \
                vec::batched_matmat2mul_transposed(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(-2), lhs.GetDim(-1), rhs.GetDim(-2), output.GetSize()); \
            } else {\
                vec::batched_matmat2mul(lhs.data_ptr<T>(), rhs.data_ptr<T>(), output.data_ptr<T>(), lhs.GetDim(-2), lhs.GetDim(-1), rhs.GetDim(-1), output.GetSize()); \
            } \
            break;
        switch(output_type) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
        return output;
    }
    throw NotImplemetedError{"Not implemented yet."};
}

}