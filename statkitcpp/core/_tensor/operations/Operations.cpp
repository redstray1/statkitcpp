#include "Operations.h"
#include "errors.h"
#include "memory_ops.h"
#include "../shape.h"
#include "Scalar.h"
#include "ScalarType.h"
#include "Sum.h"
#include "Mean.h"
#include "Var.h"
#include <functional>

namespace statkitcpp {


template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op) {
    auto output_shape = BroadcastShapes(lhs.GetShape(), rhs.GetShape());
    ScalarType output_type = PromoteTypes(lhs.GetDType(), rhs.GetDType());
    Tensor output(output_shape, output_type);

    auto lhs_shape = lhs.GetShape();
    auto rhs_shape = rhs.GetShape();

    for (size_t output_index = 0; output_index < output.GetSize(); output_index++) {
        auto out_indexes = output.GetIndexesFromFlat(output_index);
        
        std::vector<size_t> lhs_indexes(lhs.GetNDim());
        for (size_t back_idx = 0; back_idx < lhs.GetNDim(); back_idx++) {
            size_t lhs_dim = lhs.GetNDim() - back_idx - 1;
            size_t out_dim = output.GetNDim() - back_idx - 1;
            if (out_dim < 0) {
                lhs_indexes[lhs_dim] = 0;
                continue;
            }
            lhs_indexes[lhs_dim] = out_indexes[out_dim];
            if (lhs_shape[lhs_dim] == 1) {
                lhs_indexes[lhs_dim] = 0;
            }
        }

        std::vector<size_t> rhs_indexes(rhs.GetNDim());
        for (size_t back_idx = 0; back_idx < rhs.GetNDim(); back_idx++) {
            size_t rhs_dim = rhs.GetNDim() - back_idx - 1;
            size_t out_dim = output.GetNDim() - back_idx - 1;
            if (out_dim < 0) {
                rhs_indexes[rhs_dim] = 0;
                continue;
            }
            rhs_indexes[rhs_dim] = out_indexes[out_dim];
            if (rhs_shape[rhs_dim] == 1) {
                rhs_indexes[rhs_dim] = 0;
            }
        }

        size_t lhs_index = lhs.GetFlatIndex(lhs_indexes);
        size_t rhs_index = rhs.GetFlatIndex(rhs_indexes);

        if (lhs_index >= lhs.GetSize()) {
            throw OutOfRangeFlatError{lhs_index, lhs.GetSize()};
        }
        if (rhs_index >= rhs.GetSize()) {
            std::cout << "output index: ";
            for (auto& x : out_indexes) {
                std::cout << x << ' ';
            }
            std::cout << '\n';
            throw OutOfRangeFlatError{rhs_index, rhs.GetSize()};
        }
        calc(lhs.GetDataPointer(), lhs.GetDType(), lhs_index,
             rhs.GetDataPointer(), rhs.GetDType(), rhs_index,
             output.GetDataPointer(), output_type, output_index, op);
        //output.data_[output_index] = op(lhs.data_[lhs_index], rhs.data_[rhs_index]);
    }
    return output;
}

Tensor AddImpl(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha = 1) {
    return ApplyBinaryOp(lhs, rhs, std::plus());
}

Tensor SubImpl(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha = 1) {
    return ApplyBinaryOp(lhs, rhs, std::minus());
}

Tensor MulImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::multiplies());
}

Tensor DivImpl(const Tensor& lhs, const Tensor& rhs) {
    return ApplyBinaryOp(lhs, rhs, std::divides());
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
    ops::sum(arg.GetStorage(), arg.GetDType(), shape, strides, dim, keepdims, out.GetStorage());
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
    ops::mean(arg.GetStorage(), arg.GetDType(), shape, strides, dim, keepdims, out.GetStorage());
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
    ops::var(arg.GetStorage(), arg.GetDType(), shape, strides, dim, keepdims, out.GetStorage());
    return out;
}

}