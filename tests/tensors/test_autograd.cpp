#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

namespace statkitcpp {

TEST_CASE("Add autograd") {
    {
        Tensor a = Ones({2, 3});
        Tensor b = Ones({1, 3});
        a.SetRequiresGrad(true);
        b.SetRequiresGrad(true);
        auto c = a.Add(b);
        c.Backward();
    }
}
}