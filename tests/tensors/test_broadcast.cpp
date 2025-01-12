#include "../../statkitcpp/core/_tensor/tensor.h"

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

TEST_CASE("Simple broadcast") {
    {
        Tensor<float> a({1, 1});
        Tensor<float> b({1, 1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({2, 1});
        Tensor<float> b({1, 1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({2, 1});
        Tensor<double> b({1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({5, 4});
        Tensor<double> b({1});
        REQUIRE(a.BroadcastableTo(b));

        a = Tensor({5, 4});
        b = Tensor({4});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({15, 3, 5});
        Tensor<double> b({15, 1, 5});
        REQUIRE(a.BroadcastableTo(b));

        a = Tensor({15, 3, 5});
        b = Tensor({3, 5});
        REQUIRE(a.BroadcastableTo(b));

        a = Tensor({15, 3, 5});
        b = Tensor({3, 1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor a({8, 1, 6, 1});
        Tensor b({7, 1, 5});
        REQUIRE(a.BroadcastableTo(b));
    }
}

TEST_CASE("Not broadcastable") {
    {
        auto a = Tensor({3});
        auto b = Tensor({4});
        REQUIRE_FALSE(a.BroadcastableTo(b));
    }
    {
        auto a = Tensor({2, 1});
        auto b = Tensor({8, 4, 3});
        REQUIRE_FALSE(a.BroadcastableTo(b));
    }
}

}