#include "../../statkitcpp/core/_tensor/Tensor.h"

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

template <class T1, class T2>
void CheckVector(const std::vector<T1>& v1, const std::vector<T2>& v2) {
    CHECK(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); i++) {
        CHECK(v1[i] == v2[i]);
    }
}

TEST_CASE("Base info of Tensor") {
    {
        Tensor a({2, 2});
        CHECK(a.GetItemSize() == 4);
        CHECK(a.GetNBytes() == 16);
        CHECK(a.GetSize() == 4);
        CHECK(a.GetNDim() == 2);
        std::vector<uint32_t> r = {8, 4};
        CheckVector(a.GetStrides(), r);
    }
    {
        Tensor a({2, 3, 2, 2, 4});
        CHECK(a.GetItemSize() == 4);
        CHECK(a.GetNBytes() == 384);
        CHECK(a.GetSize() == 96);
        CHECK(a.GetNDim() == 5);
        std::vector<uint32_t> r = {192, 64, 32, 16, 4};
        CheckVector(a.GetStrides(), r);
    }
}

}