#include "Operations.h"
#include "Tensor.h"
#include "errors.h"
#include "memory_ops.h"
#include "../shape.h"
#include "Scalar.h"
#include "ScalarType.h"
#include "Sum.h"
#include "Mean.h"
#include "Var.h"
#include "Prod.h"
#include "Max.h"
#include "MaxDeriv.h"
#include "MinDeriv.h"
#include "Min.h"
#include <cmath>
#include <functional>
#include "function_objects.h"

namespace statkitcpp {


template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op, ScalarType out_type) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = out_type;
    if (out_type == ScalarType::Undefined) {
        output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());;
    }
    Tensor output(output_shape, output_type);

    ops::binary_op(lhs.GetDataPointer(), lhs.GetDType(), lhs.GetShape(), lhs.GetStrides(), output.GetSize(),
                   rhs.GetDataPointer(), rhs.GetDType(), rhs.GetShape(), rhs.GetStrides(),
                   output.GetDataPointer(), output.GetDType(), output.GetStrides(), op);
    return output;
}

template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Scalar& rhs, BinaryOperation op, ScalarType out_type) {
    ScalarType output_type = out_type;
    if (out_type == ScalarType::Undefined) {
        output_type = PromoteTypes(lhs.GetDType(), rhs.Type());
    }
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

Tensor EqualImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::equal(), ScalarType::Bool);
}

Tensor EqualImpl(const Tensor& lhs, const Scalar& rhs) {
    return ApplyBinaryOp(lhs, rhs, func::equal(), ScalarType::Bool);
}

Tensor CloseImpl(const Tensor& lhs, const Tensor& rhs, long double atol) {
    return ApplyBinaryOp(lhs, rhs, func::close(atol), ScalarType::Bool);
}

Tensor CloseImpl(const Tensor& lhs, const Scalar& rhs, long double atol) {
    return ApplyBinaryOp(lhs, rhs, func::close(atol), ScalarType::Bool);
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
    if (shape[dim] <= 1) {
        throw DegreesOfFreedomError{};
    }
    auto strides = arg.GetStrides();
    ops::var(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}


Tensor MaxImpl(const Tensor& arg, int dim, bool keepdims) {
    if (arg.GetNDim() == 1) {
        keepdims = true;
    }

    Tensor out(RemoveDim(arg.GetShape(), dim, keepdims), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::max(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

Tensor MaxDerivImpl(const Tensor& arg, int dim) {
    Tensor out(arg.GetShape(), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::max_deriv(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

Tensor MinImpl(const Tensor& arg, int dim, bool keepdims) {
    if (arg.GetNDim() == 1) {
        keepdims = true;
    }

    Tensor out(RemoveDim(arg.GetShape(), dim, keepdims), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::min(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

Tensor MinDerivImpl(const Tensor& arg, int dim) {
    Tensor out(arg.GetShape(), arg.GetDType());
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    auto shape = arg.GetShape();
    auto strides = arg.GetStrides();
    ops::min_deriv(arg.GetStorage(), arg.GetDType(), shape, strides, dim, out.GetStorage());
    return out;
}

}