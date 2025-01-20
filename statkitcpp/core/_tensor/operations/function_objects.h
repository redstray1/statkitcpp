#pragma once

#include <functional>
#include <cmath>

namespace statkitcpp {
namespace func {
template <class Type = void>
struct pow : public std::binary_function <Type, Type, Type> //NOLINT
{
    Type operator()(const Type& x, const Type& y) const {
        return std::pow(x, y);
    }
};

template <>
struct pow<void>
{
  template <class T, class U>
  auto operator()(T&& x, U&& y) const
    -> decltype(std::pow(std::forward<T>(x), std::forward<U>(y))) {
      return std::pow(std::forward<T>(x), std::forward<U>(y));
    }
};

template <class Type = void>
struct exp : public std::unary_function <Type, Type> //NOLINT
{
    Type operator()(const Type& x) const {
      return std::exp(x);
    }
};

template <>
struct exp<void>
{
  template <class T>
  auto operator()(T&& x) const
    -> decltype(std::exp(std::forward<T>(x))) {
      return std::exp(std::forward<T>(x));
    }
};


template <class Type = void>
struct log : public std::unary_function <Type, Type> //NOLINT
{
    Type operator()(const Type& x) const {
      return std::log(x);
    }
};

template <>
struct log<void>
{
  // template <typename T,
  //           typename std::enable_if_t<
  //               std::is_same_v<T, int16_t>, bool> = true>
  // auto operator()(T&& x) const
  //   -> decltype(std::log(std::forward<double>(static_cast<double>(x)))) {
  //     return 
  //   }

  // template <typename T,
  //           typename std::enable_if_t<
  //               !std::is_same_v<T, int16_t>, bool> = false>
  // auto operator()(T&& x) const
  //   -> decltype(std::log(std::forward<T>(x)));
  template <typename T>
  auto operator()(T&& x) const
    -> decltype(std::log(std::forward<T>(x))) {
      return std::log(std::forward<T>(x));
    }
};



template <class Type = void>
struct sqrt : public std::unary_function <Type, Type> //NOLINT
{
    Type operator()(const Type& x) const {
      return std::sqrt(x);
    }
};

template <>
struct sqrt<void>
{
  // template <typename T,
  //           typename std::enable_if_t<
  //               std::is_same_v<T, int8_t>, bool> = true>
  // auto operator()(T&& x) const
  //   -> decltype(std::sqrt(std::forward<double>(static_cast<double>(x))));
  
  // template <typename T,
  //           typename std::enable_if_t<
  //               !std::is_same_v<T, int8_t>, bool> = false>
  // auto operator()(T&& x) const
  //   -> decltype(std::sqrt(std::forward<T>(x)));

  template <typename T>
  auto operator()(T&& x) const
    -> decltype(std::sqrt(std::forward<T>(x))) {
      return std::sqrt(std::forward<T>(x));
    }
};
}
}