#include "TensorImpl.h"
#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"

namespace statkitcpp {

Tensor AddFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = rhs.GetImpl();
    auto output = AddImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor AddFunction::Forward(const Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = nullptr;
    auto output = AddImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> AddFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = grad_output;

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (size_t i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        input_grads[0] = dlhs;
        // lhs_->Backward(dlhs);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = grad_output;

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (size_t i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        input_grads[1] = drhs;
        // rhs_->Backward(drhs);
    }
    return input_grads;
}

std::string AddFunction::GetName() const {
    return "AddFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> AddFunction::GetChildren() {
    return {lhs_, rhs_};
}



Tensor SubFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = rhs.GetImpl();
    auto output = SubImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor SubFunction::Forward(const Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = nullptr;
    auto output = SubImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> SubFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = grad_output;

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (size_t i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        input_grads[0] = dlhs;
        // lhs_->Backward(dlhs, output);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = NegImpl(grad_output);

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (size_t i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        input_grads[1] = drhs;
        // rhs_->Backward(drhs, output);
    }
    return input_grads;
}

std::string SubFunction::GetName() const {
    return "SubFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> SubFunction::GetChildren() {
    return {lhs_, rhs_};
}





Tensor MulFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = rhs.GetImpl();
    auto output = MulImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor MulFunction::Forward(const Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = nullptr;
    auto output = MulImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> MulFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = MulImpl(grad_output, Tensor(rhs_));

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (size_t i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        input_grads[0] = dlhs;
        // lhs_->Backward(dlhs, output);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = MulImpl(grad_output, Tensor(lhs_));

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (size_t i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        input_grads[1] = drhs;
    }
    return input_grads;
}

std::string MulFunction::GetName() const {
    return "MulFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> MulFunction::GetChildren() {
    return {lhs_, rhs_};
}




Tensor DivFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = rhs.GetImpl();
    auto output = DivImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor DivFunction::Forward(const Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = nullptr;
    auto output = DivImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> DivFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = DivImpl(grad_output, Tensor(rhs_));

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (size_t i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        input_grads[0] = dlhs;
        // lhs_->Backward(dlhs, output);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = DivDerivImpl(Tensor(lhs_), Tensor(rhs_));
        drhs = MulImpl(grad_output, drhs);

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (size_t i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        input_grads[1] = drhs;
    }
    return input_grads;
}

std::string DivFunction::GetName() const {
    return "DivFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> DivFunction::GetChildren() {
    return {lhs_, rhs_};
}




Tensor PowFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = rhs.GetImpl();
    auto output = PowImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor PowFunction::Forward(const Tensor& lhs, const Scalar& rhs) {
    bool requires_grad = lhs.GetRequiresGrad();
    lhs_ = lhs.GetImpl();
    rhs_ = nullptr;
    auto output = PowImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> PowFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(2);
    if (lhs_->GetRequiresGrad()) {
        Tensor dlhs = MulImpl(grad_output, PowDerivImpl(Tensor(lhs_), Tensor(rhs_)));

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = lhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            dlhs = SumImpl(dlhs, 0, false);
        }
        auto shape = lhs_->GetShape();
        for (size_t i = 0; i < lhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                dlhs = SumImpl(dlhs, i, true);
            }
        }
        input_grads[0] = dlhs;
        // lhs_->Backward(dlhs, output);
    }
    if (rhs_ != nullptr && rhs_->GetRequiresGrad()) {
        auto drhs = ExpDerivImpl(Tensor(lhs_), Tensor(rhs_));
        drhs = MulImpl(grad_output, drhs);

        auto grad_dim = grad_output.GetNDim();
        auto in_dim = rhs_->GetNDim();
        for (int i = 0; i < static_cast<int>(grad_dim - in_dim); i++) {
            drhs = SumImpl(drhs, 0, false);
        }
        auto shape = rhs_->GetShape();
        for (size_t i = 0; i < rhs_->GetNDim(); i++) {
            if (shape[i] == 1) {
                drhs = SumImpl(drhs, i, true);
            }
        }
        input_grads[1] = drhs;
    }
    return input_grads;
}

std::string PowFunction::GetName() const {
    return "PowFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> PowFunction::GetChildren() {
    return {lhs_, rhs_};
}

}