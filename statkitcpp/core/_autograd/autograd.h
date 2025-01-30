#ifndef AUTOGRAD_HEADER_H
#define AUTOGRAD_HEADER_H
#include "../include_fwd.h"
#include <vector>
#include <memory>
#include <optional>
#include "Node.h"
#include <string>

namespace statkitcpp {

class GradFunction {
public:
    virtual ~GradFunction() {};
    virtual std::vector<Tensor> Backward(const Tensor& grad_output) = 0;
    virtual std::vector<std::shared_ptr<TensorImpl>> GetChildren() = 0;
    virtual std::string GetName() const = 0;
};

void RunBackward(const std::shared_ptr<Node>& root, std::optional<Tensor> grad_output, bool retain_graph=false);

}
#endif