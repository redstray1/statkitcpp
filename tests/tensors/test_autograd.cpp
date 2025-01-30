#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"
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

TEST_CASE("Complex autograd") {
    {
        Tensor a = Arange(0, 5);
        Tensor b = Arange(0, 5).Reshape({5, 1});
        b.SetRequiresGrad(true);
        a.SetRequiresGrad(true);
        auto c = a.Add(b);
        REQUIRE(a.GetRequiresGrad());
        REQUIRE(b.GetRequiresGrad());
        REQUIRE(c.GetRequiresGrad());
        c.Backward();
        REQUIRE_THROWS(b.GetGrad());
        std::cout << (a.GetGrad()).ToString() << std::endl;
    }
    {
        Tensor a = Arange(0, 5);
        a.SetRequiresGrad(true);
        auto c = a.Exp();
        c = c.Mean();
        c.Backward();
        std::cout << (a.GetGrad()).ToString() << std::endl;
    }
}

}