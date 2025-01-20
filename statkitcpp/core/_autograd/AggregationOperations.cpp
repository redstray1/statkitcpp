#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"

namespace statkitcpp {


Tensor SumFunction::Forward(Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = SumImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void SumFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string SumFunction::GetName() const {
    return "SumFunction()";
}


Tensor ProdFunction::Forward(Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = ProdImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void ProdFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string ProdFunction::GetName() const {
    return "ProdFunction()";
}


Tensor MeanFunction::Forward(Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = MeanImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}



void MeanFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string MeanFunction::GetName() const {
    return "MeanFunction()";
}



Tensor VarFunction::Forward(Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = VarImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void VarFunction::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
}

std::string VarFunction::GetName() const {
    return "VarFunction()";
}

}