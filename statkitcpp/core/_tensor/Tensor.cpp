#include "Tensor.h"
#include "../_autograd/operations.h"
#include "../operations/Operations.h"
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
        grad_output = Ones(GetShape());
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

// Tensor class implementation END-----------------------------------------------------

}