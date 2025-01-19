#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"

#include <catch2/catch_test_macros.hpp>
#include <initializer_list>
#include <iostream>

namespace statkitcpp {

template <class T1, class T2>
void CheckVector(const std::vector<T1>& v1, const std::vector<T2>& v2) {
    CHECK(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); i++) {
        CHECK(v1[i] == v2[i]);
    }
}

TEST_CASE("Sum aggregate operation") {
    {
        auto a = Ones({2, 4});
        auto b = a.Sum();
        CHECK(b.GetNDim() == 1);
        CHECK(b.GetSize() == 2);
        // std::vector<float> r = {4, 4};
        // CheckVector(b.GetData(), r);
    }
    {
        auto a = Ones({2, 4});
        auto b = a.Sum(0);
        CHECK(b.GetNDim() == 1);
        CHECK(b.GetSize() == 4);
        // std::vector<float> r = {2, 2, 2, 2};
        // CheckVector(b.GetData(), r);
    }
    {
        auto a = Ones({2, 3, 2});
        auto b = a.Sum(2);
        CHECK(b.GetNDim() == 2);
        CHECK(b.GetSize() == 6);
        // std::vector<float> r = {2, 2, 2, 2, 2, 2};
        // CheckVector(b.GetData(), r);
    }
    {
        auto a = Ones({2, 3, 2}, kFloat64);
        auto b = a.Sum();
        CHECK(b.GetNDim() == 2);
        CHECK(b.GetSize() == 6);
    }
}

TEST_CASE("Var aggregate operation") {
    {
        // auto a = Tensor({1, 2}, {1, 2});
        // auto b = a.Var();
        // std::vector<float> r = {0.25};
        // CheckVector(b.GetData(), r);
    }
}

}