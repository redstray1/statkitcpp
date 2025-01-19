#ifndef AUTOGRAD_HEADER_H
#define AUTOGRAD_HEADER_H
#include "../include_fwd.h"

namespace statkitcpp {

class GradFunction {
public:
    virtual ~GradFunction() {};
    virtual Tensor Backward(const Tensor& grad_output, const Tensor& output) = 0;
};

}
#endif