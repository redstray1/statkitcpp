#include "Tensor.h"
#include "TensorCreationOps.h"
#include "datatypes.h"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <optional>

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

TEST_CASE("Simple autograd") {
    {
        Tensor a = Arange(0, 5);
        a.SetRequiresGrad(true);
        auto c = a.Exp();
        c = c.Mean();
        c.Backward();
        // std::cout << (a.GetGrad()).ToString() << std::endl;
    }
    {
        Tensor a = Full({1, 1}, 0.697328);
        Tensor b = Full({1}, 0.580362);
        a.SetRequiresGrad(true);
        b.SetRequiresGrad(true);
        auto c = a / b;
        c.Backward();
    }
    {
        Tensor a = Full({100, 20, 5, 2}, 4);
        a.SetRequiresGrad(true);
        auto c = a.Sum(3);
        c.Backward();
    }
    {
        Tensor a = Arange(1, 4);
        a.SetRequiresGrad(true);
        auto c = a.Var();
        c.Backward();
    }
    {
        Tensor a = Ones({5, 5}) * Arange(0, 5);
        a.SetRequiresGrad(true);
        auto b = a.Max(-1);
        b.Backward(std::nullopt, std::nullopt, true);
    }
    {
        Tensor a = Ones({1, 3, 2});
        Tensor c = Full({1}, 2);
        a.SetRequiresGrad(true);
        auto b = a * c;
        b.Backward();
        // std::cout << (a.GetGrad()).ToString();
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
    }
    
}

}