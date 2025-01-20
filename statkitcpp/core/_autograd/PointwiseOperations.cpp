#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"


namespace statkitcpp {

Tensor NegFunction::Forward(Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = NegImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void NegFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string NegFunction::GetName() const {
    return "NegFunction()";
}



Tensor ExpFunction::Forward(Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = ExpImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void ExpFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string ExpFunction::GetName() const {
    return "ExpFunction()";
}


Tensor LogFunction::Forward(Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = LogImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void LogFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string LogFunction::GetName() const {
    return "LogFunction()";
}

Tensor SqrtFunction::Forward(Tensor& arg) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = SqrtImpl(arg);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void SqrtFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string SqrtFunction::GetName() const {
    return "SqrtFunction()";
}

}