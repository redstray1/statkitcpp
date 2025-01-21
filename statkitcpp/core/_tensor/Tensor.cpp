#include "Tensor.h"
#include "../_autograd/operations.h"
#include "../operations/Operations.h"
#include "Storage.h"
#include "TensorCreationOps.h"
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

Tensor::Tensor(const Storage& storage,
               const std::vector<size_t>& shape,
               ScalarType dtype,
               bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(storage, shape, dtype);
    requires_grad_ = requires_grad;
}

Tensor::Tensor(const Tensor& other) {
    impl_ = other.impl_;
    grad_fn = other.grad_fn;
    requires_grad_ = other.requires_grad_;
}

Tensor::Tensor(Tensor&& other) {
    impl_ = std::move(other.impl_);
    grad_fn = std::move(other.grad_fn);
    requires_grad_ = std::move(other.requires_grad_);
}

Tensor& Tensor::operator=(const Scalar& other) && {
    *this = Full(GetShape(), other, other.Type());
    return *this;
}

// Constructors END------------------------------------------------------------

std::string Tensor::GetTypeName() const {
    return impl_->GetTypeName();
}

std::string Tensor::ToString() const {
    auto repr = impl_->ToString();
    if (grad_fn != nullptr) {
        repr += ", grad_fn=";
        repr += grad_fn->GetName();
    }
    repr += ")";
    return repr;
}

//------------------------------------------------------------------------------------
//Public methods----------------------------------------------------------------------
//------------------------------------------------------------------------------------

bool Tensor::BroadcastableTo(const Tensor& other) {
    return IsBroadcastable(GetShape(), other.GetShape());
}

void Tensor::Backward(std::optional<Tensor> grad_output, std::optional<Tensor> output, bool retain_graph) {
    if (!requires_grad_) {
        return;
    }
    if (!grad_output.has_value()) {
        grad_output = Ones(GetShape());
    }
    if (grad == nullptr) {
        grad = std::make_shared<Tensor>(Zeros(GetShape()));
    }
    *grad = AddImpl(*grad, grad_output.value());
    if (grad_fn != nullptr) {
        grad_fn->Backward(*grad, *this);
    }
}

Tensor Tensor::Reshape(const std::vector<size_t>& shape) {
    auto op = std::make_shared<ReshapeOperation>();
    auto result = op->Forward(*this, shape);
    result.grad_fn = op;
    return result;
}

// Tensor class implementation END-----------------------------------------------------

}