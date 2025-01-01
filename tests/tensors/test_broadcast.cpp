#include "../../statkitcpp/core/tensor.h"

#include <catch2/catch_test_macros.hpp>
#include <iostream>

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
        Tensor<float> b({1});
        REQUIRE(a.BroadcastableTo(b));
    }
}
}