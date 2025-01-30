#pragma once

#include <memory>
#include "include_fwd.h"

namespace statkitcpp {

struct Node {
    std::weak_ptr<TensorImpl> impl;
    std::unique_ptr<Tensor> temp_grad = nullptr;
    void AddGrad(const Tensor& to_add);
};

}