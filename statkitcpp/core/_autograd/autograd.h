#ifndef AUTOGRAD_HEADER_H
#define AUTOGRAD_HEADER_H
#include "../include_fwd.h"
#include <string>

namespace statkitcpp {

class GradFunction {
public:
    virtual ~GradFunction() {};
    virtual void Backward(const Tensor& grad_output, const Tensor& output) = 0;
    virtual std::string GetName() const = 0;
};

void RunBackward(Tensor* start);

}
#endif