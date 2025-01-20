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
    *grad = AddImpl(*grad, grad_output.value(), 1);
    if (grad_fn != nullptr) {
        grad_fn->Backward(*grad, *this);
    }
}

//Aggregation operations
Tensor Tensor::Sum(int dim, bool keepdims) {
    auto op = std::make_shared<SumFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Prod(int dim, bool keepdims) {
    auto op = std::make_shared<ProdFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Mean(int dim, bool keepdims) {
    auto op = std::make_shared<MeanFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Var(int dim, bool keepdims) {
    auto op = std::make_shared<VarFunction>();
    auto result = op->Forward(*this, dim, keepdims);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Add(Tensor& other, const Scalar& alpha) {
    auto op = std::make_shared<AddFunction>();
    auto result = op->Forward(*this, other, alpha);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Sub(Tensor& other, const Scalar& alpha) {
    auto op = std::make_shared<SubFunction>();
    auto result = op->Forward(*this, other, alpha);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Mul(Tensor& other) {
    auto op = std::make_shared<MulFunction>();
    auto result = op->Forward(*this, other);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Div(Tensor& other) {
    auto op = std::make_shared<DivFunction>();
    auto result = op->Forward(*this, other);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Pow(Tensor& other) {
    auto op = std::make_shared<PowFunction>();
    auto result = op->Forward(*this, other);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Neg() {
    auto op = std::make_shared<NegFunction>();
    auto result = op->Forward(*this);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Exp() {
    auto op = std::make_shared<ExpFunction>();
    auto result = op->Forward(*this);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Log() {
    auto op = std::make_shared<LogFunction>();
    auto result = op->Forward(*this);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Sqrt() {
    auto op = std::make_shared<SqrtFunction>();
    auto result = op->Forward(*this);
    result.grad_fn = op;
    return result;
}

Tensor Tensor::Reshape(const std::vector<size_t>& shape) {
    auto op = std::make_shared<ReshapeOperation>();
    auto result = op->Forward(*this, shape);
    result.grad_fn = op;
    return result;
}

// Tensor class implementation END-----------------------------------------------------

}