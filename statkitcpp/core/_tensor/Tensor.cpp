#include "Tensor.h"
#include "../_autograd/operations.h"
#include "ScalarType.h"
#include "shape.h"
#include "TensorImpl.h"
#include <cstddef>
#include <cassert>
#include <sys/types.h>
#include <memory>

namespace statkitcpp {

// Tensor class implementation START-----------------------------------------------------
// Constructors START------------------------------------------------------------

Tensor::Tensor() {
    impl_ = nullptr;
}

Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype, bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(shape,  dtype);
    requires_grad_ = requires_grad;
}

Tensor::Tensor(void* data, 
               const std::vector<size_t>& shape,
               ScalarType dtype,
               bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(data, shape, dtype);
    requires_grad_ = requires_grad;
}

Tensor::Tensor(const Tensor& other) {
    impl_ = other.impl_;
    requires_grad_ = other.requires_grad_;
}

Tensor::Tensor(Tensor&& other) {
    impl_ = std::move(other.impl_);
    requires_grad_ = std::move(other.requires_grad_);
}

// Constructors END------------------------------------------------------------

std::string Tensor::GetTypeName() const {
    return impl_->GetTypeName();
}

// size_t Tensor::GetFlatIndex(const std::vector<size_t>& indexes) const {
//     return impl_->GetFlatIndex(indexes);
// }

// template <typename T>
// std::vector<size_t> Tensor<T>::GetIndexesFromFlat(size_t flat_index) const {
//     std::vector<size_t> indexes(shape_.size(), 0);
//     if (flat_index >= size_) {
//         throw OutOfRangeFlatError{flat_index, size_};
//     }
//     for (size_t i = 0; i < shape_.size(); ++i) {
//         indexes[i] = flat_index / strides_[i];
//         flat_index %= strides_[i];
//     }
//     return indexes;
// }

// template <typename T>
// template <class BinaryOperation>
// Tensor<T> Tensor<T>::ApplyBroadcastOp(const Tensor<T>& lhs, const Tensor<T>& rhs,
//                                       BinaryOperation op) {
//     if (!IsBroadcastable(lhs.shape_, rhs.shape_)) {
//         throw BroadcastError{lhs.ShapeToString(), rhs.ShapeToString()};
//     }                                   
//     std::vector<size_t> output_shape;
//     for (int i = std::max(lhs.shape_.size(), rhs.shape_.size()) - 1; i >= 0; i--) {
//         size_t lhs_dim = i < static_cast<int>(lhs.shape_.size()) ? lhs.shape_[i] : 1;
//         size_t rhs_dim = i < static_cast<int>(rhs.shape_.size()) ? rhs.shape_[i] : 1;
//         output_shape.push_back(std::max(lhs_dim, rhs_dim));
//     }
//     std::reverse(output_shape.begin(), output_shape.end());
    
//     Tensor<T> output(output_shape);

//     for (size_t i = 0; i < output.size_; ++i) {
//         std::vector<size_t> output_indexes = output.GetIndexesFromFlat(i);
//         std::vector<size_t> lhs_indexes(output_indexes);
//         std::vector<size_t> rhs_indexes(output_indexes);
//         for (int j = static_cast<int>(lhs.shape_.size()) - 1; j >= 0; --j) {
//             if (j >= output_shape.size() || lhs.shape_[j] == 1) {
//                 lhs_indexes[j] = 0;
//             }
//         }
//         for (int j = static_cast<int>(rhs.shape_.size()) - 1; j >= 0; --j) {
//             if (j >= output_shape.size() || rhs.shape_[j] == 1) {
//                 rhs_indexes[j] = 0;
//             }
//         }
//         size_t lhs_flat_index = lhs.GetFlatIndex(lhs_indexes);
//         size_t rhs_flat_index = rhs.GetFlatIndex(rhs_indexes);

//         output.data_[i] = op(lhs.data_[lhs_flat_index], rhs.data_[rhs_flat_index]);
//     } 
//     return output;
// }

//------------------------------------------------------------------------------------
//Public methods----------------------------------------------------------------------
//------------------------------------------------------------------------------------

bool Tensor::BroadcastableTo(const Tensor& other) {
    return IsBroadcastable(GetShape(), other.GetShape());
}

Tensor Tensor::Sum(int dim, bool keepdims) const {
    auto op = std::make_shared<SumFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Mean(int dim, bool keepdims) const {
    auto op = std::make_shared<MeanFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Var(int dim, bool keepdims) const {
    auto op = std::make_shared<VarFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Add(const Tensor& other, const Scalar& alpha) const {
    auto op = std::make_shared<AddFunction>();
    auto result = op->Forward(*this, other, alpha);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Sub(const Tensor& other, const Scalar& alpha) const {
    auto op = std::make_shared<SubFunction>();
    auto result = op->Forward(*this, other, alpha);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Mul(const Tensor& other) const {
    auto op = std::make_shared<MulFunction>();
    auto result = op->Forward(*this, other);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Div(const Tensor& other) const {
    auto op = std::make_shared<DivFunction>();
    auto result = op->Forward(*this, other);
    result.grad_fn = op;
    return result;
}

// Tensor class implementation END-----------------------------------------------------

}