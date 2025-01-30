#include "Operations.h"
#include "errors.h"
#include "memory_ops.h"
#include "../shape.h"
#include "Scalar.h"
#include "ScalarType.h"
#include "Sum.h"
#include "Mean.h"
#include "Var.h"
#include "Prod.h"
#include <cmath>
#include <functional>
#include "function_objects.h"

namespace statkitcpp {


template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(output_shape, output_type);

    ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(), output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output.GetDType(), output.GetStrides(), op);
    return output;
}

template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Scalar& rhs, BinaryOperation op) {
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.Type());
    Tensor output(lhs.GetShape(), output_type);

    ops::scalar_op(lhs.GetDataPointer(), lhs.GetDType(), output.GetSize(),
                   rhs, output.GetDataPointer(), output_type, op);
    return output;
}

Tensor AddImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::plus());
}

Tensor AddImpl(const Tensor& lhs, const Scalar& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::plus());
}

Tensor SubImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::minus());
}

Tensor SubImpl(const Tensor& lhs, const Scalar& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::minus());
}

Tensor MulImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::multiplies());
}

Tensor MulImpl(const Tensor& lhs, const Scalar& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::multiplies());
}

Tensor DivImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::divides());
}

Tensor DivImpl(const Tensor& lhs, const Scalar& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::divides());
}

Tensor PowImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::pow());
}

Tensor PowImpl(const Tensor& lhs, const Scalar& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::pow());
}

Tensor DivDerivImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::div_deriv());
}

Tensor PowDerivImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::pow_deriv());
}

Tensor ExpDerivImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::exp_deriv());
}

Tensor SumImpl(const Tensor &arg, int dim, bool keepdims) {

    if (arg.GetNDim() == 1) {
        keepdims = true;
    }
    Tensor out(RemoveDim(arg.GetShape(), dim, keepdims), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::sum(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}


Tensor ProdImpl(const Tensor &arg, int dim, bool keepdims) {

    if (arg.GetNDim() == 1) {
        keepdims = true;
    }
    Tensor out(RemoveDim(arg.GetShape(), dim, keepdims), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::prod(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

Tensor MeanImpl(const Tensor &arg, int dim, bool keepdims) {

    if (arg.GetNDim() == 1) {
        keepdims = true;
    }

    Tensor out(RemoveDim(arg.GetShape(), dim, keepdims), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::mean(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

Tensor VarImpl(const Tensor &arg, int dim, bool keepdims) {

    if (arg.GetNDim() == 1) {
        keepdims = true;
    }

    Tensor out(RemoveDim(arg.GetShape(), dim, keepdims), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::var(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

}