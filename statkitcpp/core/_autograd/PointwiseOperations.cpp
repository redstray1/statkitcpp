#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"


namespace statkitcpp {

Tensor NegFunction::Forward(const Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = NegImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> NegFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg = NegImpl(grad_output);
        dargs[0] = darg;
    }
    return dargs;
}

std::string NegFunction::GetName() const {
    return "NegFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> NegFunction::GetChildren() {
    return {arg_};
}


Tensor ExpFunction::Forward(const Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = ExpImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> ExpFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg = ExpImpl(Tensor(arg_));
        darg = MulImpl(darg, grad_output);
        dargs[0] = darg;
    }
    return dargs;
}

std::string ExpFunction::GetName() const {
    return "ExpFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> ExpFunction::GetChildren() {
    return {arg_};
}




Tensor LogFunction::Forward(const Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = LogImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> LogFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg = ReciprocalImpl(Tensor(arg_));
        darg = MulImpl(darg, grad_output);
        dargs[0] = darg;
    }
    return dargs;
}

std::string LogFunction::GetName() const {
    return "LogFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> LogFunction::GetChildren() {
    return {arg_};
}



Tensor SqrtFunction::Forward(const Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = SqrtImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> SqrtFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg = SqrtDerivImpl(Tensor(arg_));
        darg = MulImpl(darg, grad_output);
        dargs[0] = darg;
    }
    return dargs;
}

std::string SqrtFunction::GetName() const {
    return "SqrtFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> SqrtFunction::GetChildren() {
    return {arg_};
}




}