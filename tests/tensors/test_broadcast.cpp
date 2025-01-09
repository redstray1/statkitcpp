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
}
}