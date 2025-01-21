#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"

namespace statkitcpp {

Tensor AddFunction::Forward(Tensor& lhs, Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = &rhs;
    auto output = AddImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor AddFunction::Forward(Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = nullptr;
    auto output = AddImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void AddFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = grad_output;

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (int i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        lhs_->Backward(dlhs, output);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = grad_output;

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (int i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        rhs_->Backward(drhs, output);
    }
}

std::string AddFunction::GetName() const {
    return "AddFunction()";
}



Tensor SubFunction::Forward(Tensor& lhs, Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = &rhs;
    auto output = SubImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor SubFunction::Forward(Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = nullptr;
    auto output = SubImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void SubFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = grad_output;

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (int i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        lhs_->Backward(dlhs, output);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = NegImpl(grad_output);

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (int i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        rhs_->Backward(drhs, output);
    }
}

std::string SubFunction::GetName() const {
    return "SubFunction()";
}



Tensor MulFunction::Forward(Tensor& lhs, Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = &rhs;
    auto output = MulImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor MulFunction::Forward(Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = nullptr;
    auto output = MulImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void MulFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string MulFunction::GetName() const {
    return "MulFunction()";
}



Tensor DivFunction::Forward(Tensor& lhs, Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = &rhs;
    auto output = DivImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor DivFunction::Forward(Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = nullptr;
    auto output = DivImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void DivFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string DivFunction::GetName() const {
    return "DivFunction()";
}

Tensor PowFunction::Forward(Tensor& lhs, Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = &rhs;
    auto output = PowImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor PowFunction::Forward(Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = &lhs;
    rhs_ = nullptr;
    auto output = PowImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void PowFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string PowFunction::GetName() const {
    return "PowFunction()";
}
}