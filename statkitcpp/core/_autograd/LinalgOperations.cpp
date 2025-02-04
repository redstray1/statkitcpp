#include "TensorImpl.h"
#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"


namespace statkitcpp {

Tensor DotFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    if (requires_grad) {
        lhs_ = lhs.GetImpl();
        rhs_ = rhs.GetImpl();
    }
    auto output = DotImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> DotFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = MulImpl(grad_output, Tensor(rhs_));
        input_grads[0] = dlhs;
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = MulImpl(grad_output, Tensor(rhs_));
        input_grads[1] = drhs;
    }
    return input_grads;
}

std::string DotFunction::GetName() const {
    return "DotFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> DotFunction::GetChildren() {
    return {lhs_, rhs_};
}





Tensor MatMulFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    if (requires_grad) {
        lhs_ = lhs.GetImpl();
        rhs_ = rhs.GetImpl();
    }
    auto output = MatMulImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> MatMulFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = MatMulImpl(grad_output, Tensor(rhs_), true);
        input_grads[0] = dlhs;
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        //WRONG - FIX IT AFTER TRANSPOSE IMPLEMENTATION
        auto drhs = MatMulImpl(Tensor(lhs_), grad_output, true);
        input_grads[1] = drhs;
    }
    return input_grads;
}

std::string MatMulFunction::GetName() const {
    return "MatMulFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> MatMulFunction::GetChildren() {
    return {lhs_, rhs_};
}

}