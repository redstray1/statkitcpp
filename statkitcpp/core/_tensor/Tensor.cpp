#include "Tensor.h"
#include "../_autograd/operations.h"
#include "../operations/Operations.h"
#include "tensor_cast.h"
#include "Storage.h"
#include "TensorCreationOps.h"
#include "ScalarType.h"
#include "autograd.h"
#include "errors.h"
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
    impl_ = std::make_shared<TensorImpl>(shape,  dtype, requires_grad);
}

Tensor::Tensor(void* data, 
               const std::vector<size_t>& shape,
               ScalarType dtype,
               bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(data, shape, dtype, requires_grad);
}

Tensor::Tensor(const Storage& storage,
               const std::vector<size_t>& shape,
               ScalarType dtype,
               bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(storage, shape, dtype, requires_grad);
}

Tensor::Tensor(const Tensor& other) {
    impl_ = other.impl_;
}

Tensor::Tensor(Tensor&& other) {
    impl_ = std::move(other.impl_);
}

Tensor& Tensor::operator=(const Scalar& other) {
    *this = Full(GetShape(), other, other.Type());
    return *this;
}

// Constructors END------------------------------------------------------------

std::string Tensor::GetTypeName() const {
    return impl_->GetTypeName();
}

// void CopyFrom(Tensor other) {
//     auto 
// }

//------------------------------------------------------------------------------------
//Public methods----------------------------------------------------------------------
//------------------------------------------------------------------------------------

std::string Tensor::ToString() const {
    auto repr = impl_->ToString();
    return repr;
}

Tensor Tensor::ToType(ScalarType type) const {
    if (GetDType() == type) {
        return *this;
    } else {
        if (CanCast(GetDType(), type)) {
            Tensor result(GetShape(), type, GetRequiresGrad());
            ops::copy(GetDataPointer(), GetDType(), result.GetDataPointer(), type, GetSize());
            return result;
        } else {
            throw TypeCastError{GetDTypeName(GetDType()), GetDTypeName(type)};
        }
    }
}

bool Tensor::BroadcastableTo(const Tensor& other) {
    return IsBroadcastable(GetShape(), other.GetShape());
}

void Tensor::Backward(std::optional<Tensor> grad_output, [[maybe_unused]]std::optional<Tensor> output, bool retain_graph)& {
    if (!GetRequiresGrad()) {
        return;
    }
    if (!grad_output.has_value()) {
        grad_output = Ones(GetShape(), GetDType());
    }
    auto node = impl_->GetAutogradNode();
    assert(node == impl_->autograd_node_);
    RunBackward(node, grad_output, retain_graph);
}

Tensor Tensor::Reshape(const std::vector<size_t>& shape) {
    auto op = std::make_shared<ReshapeOperation>();
    auto result = op->Forward(*this, shape);
    result.impl_->grad_fn = op;
    return result;
}

Tensor Tensor::Transpose(int dim0, int dim1) {
    auto op = std::make_shared<TransposeOperation>();
    auto result = op->Forward(*this, dim0, dim1);
    result.impl_->grad_fn = op;
    return result;
}

Tensor Tensor::Index(const std::vector<TensorIndex>& indices) const {
    auto op = std::make_shared<IndexingOperation>();
    auto result = op->Forward(*this, indices);
    result.impl_->grad_fn = op;
    return result;
}

Tensor& Tensor::IndexPut(const std::vector<TensorIndex>& indices, const Tensor& other) {
    return IndexingPutImpl(*this, indices, other);
}

Tensor& Tensor::IndexPut(const std::vector<TensorIndex>& indices, const Scalar& scalar) {
    return IndexingPutImpl(*this, indices, scalar);
}

// Tensor class implementation END-----------------------------------------------------

}