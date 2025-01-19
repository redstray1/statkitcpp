#pragma once

#include <array>

namespace statkitcpp {

template <typename V, typename... T>
inline constexpr auto ArrayOf(T&&... t) -> std::array<V, sizeof...(T)> {
    return {{std::forward<T>(t)...}};
}

}