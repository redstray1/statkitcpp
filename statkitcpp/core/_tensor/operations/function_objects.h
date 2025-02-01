#pragma once

#include <cstdlib>
#include <functional>
#include <cmath>
#include "config.h"
#include <type_traits>

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
  template <typename T>
  auto operator()(T&& x) const
    -> decltype(std::sqrt(std::forward<T>(x))) {
      return std::sqrt(std::forward<T>(x));
    }
};



template <class Type = void>
struct reciprocal : public std::unary_function <Type, Type> //NOLINT
{
    Type operator()(const Type& x) const {
      return static_cast<Type>(1) / x;
    }
};

template <>
struct reciprocal<void>
{
  template <typename T>
  auto operator()(T&& x) const
    -> decltype(1.0 /std::forward<T>(x)) {
      return 1.0 /std::forward<T>(x);
    }
};





template <class Type = void>
struct sqrt_deriv : public std::unary_function <Type, Type> //NOLINT
{
    Type operator()(const Type& x) const {
      return 1.0 / (std::sqrt(x) * static_cast<Type>(2));
    }
};

template <>
struct sqrt_deriv<void>
{
  template <typename T>
  auto operator()(T&& x) const
    -> decltype(1.0 / (std::sqrt(std::forward<T>(x)) * 2)) {
      return 1.0 / (std::sqrt(std::forward<T>(x)) * 2);
    }
};


template <class Type = void>
struct div_deriv : public std::binary_function <Type, Type, Type> //NOLINT
{
    Type operator()(const Type& x, const Type& y) const {
        return -x / (y * y);
    }
};

template <>
struct div_deriv<void>
{
  template <class T, class U>
  auto operator()(T&& x, U&& y) const
    -> decltype(-std::forward<T>(x) / (std::forward<U>(y) * std::forward<U>(y))) {
      return -std::forward<T>(x) / (std::forward<U>(y) * std::forward<U>(y));
    }
};



template <class Type = void>
struct pow_deriv : public std::binary_function <Type, Type, Type> //NOLINT
{
    Type operator()(const Type& x, const Type& y) const {
        return y * std::pow(x, y - 1);
    }
};

template <>
struct pow_deriv<void>
{
  template <class T, class U>
  auto operator()(T&& x, U&& y) const
    -> decltype(std::forward<U>(y) * std::pow(std::forward<T>(x), std::forward<U>(y) - 1)) {
      return std::forward<U>(y) * std::pow(std::forward<T>(x), std::forward<U>(y) - 1);
    }
};


template <class Type = void>
struct exp_deriv : public std::binary_function <Type, Type, Type> //NOLINT
{
    Type operator()(const Type& x, const Type& y) const {
        return std::pow(x, y) * std::log(x);
    }
};

template <>
struct exp_deriv<void>
{
  template <class T, class U>
  auto operator()(T&& x, U&& y) const
    -> decltype(std::pow(std::forward<T>(x), std::forward<U>(y)) * std::log(std::forward<T>(x))) {
      return std::pow(std::forward<T>(x), std::forward<U>(y)) * std::log(std::forward<T>(x));
    }
};


template <class Type = void>
struct equal : public std::binary_function <Type, Type, Type> //NOLINT
{
    Type operator()(const Type& x, const Type& y) const {
        return (x == y);
    }
};

template <>
struct equal<void>
{
  template <class T, class U>
  auto operator()(T&& x, U&& y) const
    -> decltype(std::forward<T>(x) == std::forward<U>(y)) {
      return std::forward<T>(x) == std::forward<U>(y);
    }
};


template <class Type = void>
struct close : public std::binary_function <Type, Type, Type> //NOLINT
{
    long double atol = 1e-9;
    close() {}
    close(long double atol) : atol(atol) {}
    Type operator()(const Type& x, const Type& y) const {
        return abs(x - y) < atol;
    }
};

template <>
struct close<void>
{
  long double atol = kMachinePrecision;
  close() {}
  close(long double atol) : atol(atol) {}
  template <class T, class U>
  auto operator()(T&& x, U&& y) const
    -> decltype(abs(std::forward<T>(x) - std::forward<U>(y)) < atol) {
      return abs(std::forward<T>(x) - std::forward<U>(y)) < atol;
    }
};



}
}