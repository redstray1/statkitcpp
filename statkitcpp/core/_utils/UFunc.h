#pragma once

#include <functional>
#include <array>
#include "ScalarType.h"

namespace statkitcpp {

#define DEFINE_TYPES_OF_BINARY_ARGS(_) \
    _(int8_t, int8_t, int8_t) \
    _(int8_t, int16_t, int16_t) \
    _(int8_t, int, int) \
    _(int8_t, int64_t, int64_t) \
    _(int8_t, float, float) \
    _(int8_t, double, double) \
    _(int8_t, bool, int8_t) \
    _(int16_t, int8_t, int16_t) \
    _(int16_t, int16_t, int16_t) \
    _(int16_t, int, int) \
    _(int16_t, int64_t, int64_t) \
    _(int16_t, float, float) \
    _(int16_t, double, double) \
    _(int16_t, bool, int16_t) \
    _(int, int8_t, int) \
    _(int, int16_t, int) \
    _(int, int, int) \
    _(int, int64_t, int64_t) \
    _(int, float, float) \
    _(int, double, double) \
    _(int, bool, int) \
    _(int64_t, int8_t, int64_t) \
    _(int64_t, int16_t, int64_t) \
    _(int64_t, int, int64_t) \
    _(int64_t, int64_t, int64_t) \
    _(int64_t, float, float) \
    _(int64_t, double, double) \
    _(int64_t, bool, int64_t) \
    _(float, int8_t, float) \
    _(float, int16_t, float) \
    _(float, int, float) \
    _(float, int64_t, float) \
    _(float, float, float) \
    _(float, double, double) \
    _(float, bool, float) \
    _(double, int8_t, double) \
    _(double, int16_t, double) \
    _(double, int, double) \
    _(double, int64_t, double) \
    _(double, float, double) \
    _(double, double, double) \
    _(double, bool, double) \
    _(bool, int8_t, int8_t) \
    _(bool, int16_t, int16_t) \
    _(bool, int, int) \
    _(bool, int64_t, int64_t) \
    _(bool, float, float) \
    _(bool, double, double) \
    _(bool, bool, bool)


#define DEFINE_FUNC_ARRAY(funcname, funcnameT, ...) \
typedef void (*funcname##_type)(__VA_ARGS__); \
static constexpr std:: \
    array<std::array<funcname##_type, kDType2Index.size()>, kDType2Index.size()> \
      kFuncsLookup = {{ \
      /*        ki1  ki2  ki4  ki8  kf4  kf8  kb1*/ \
      /* i1 */ {int8_t_int8_t_##funcnameT,  int8_t_int16_t_##funcnameT,  int8_t_int_##funcnameT,  int8_t_int64_t_##funcnameT,  int8_t_float_##funcnameT,  int8_t_double_##funcnameT,  int8_t_bool_##funcnameT}, \
      /* i2 */ {int16_t_int8_t_##funcnameT, int16_t_int16_t_##funcnameT, int16_t_int_##funcnameT, int16_t_int64_t_##funcnameT, int16_t_float_##funcnameT, int16_t_double_##funcnameT, int16_t_bool_##funcnameT}, \
      /* i4 */ {int_int8_t_##funcnameT, int_int16_t_##funcnameT, int_int_##funcnameT, int_int64_t_##funcnameT, int_float_##funcnameT, int_double_##funcnameT, int_bool_##funcnameT}, \
      /* i8 */ {int64_t_int8_t_##funcnameT, int64_t_int16_t_##funcnameT, int64_t_int_##funcnameT, int64_t_int64_t_##funcnameT, int64_t_float_##funcnameT, int64_t_double_##funcnameT, int64_t_bool_##funcnameT}, \
      /* f4 */ {float_int8_t_##funcnameT, float_int16_t_##funcnameT, float_int_##funcnameT, float_int64_t_##funcnameT, float_float_##funcnameT, float_double_##funcnameT, float_bool_##funcnameT}, \
      /* f8 */ {double_int8_t_##funcnameT, double_int16_t_##funcnameT, double_int_##funcnameT, double_int64_t_##funcnameT, double_float_##funcnameT, double_double_##funcnameT, double_bool_##funcnameT}, \
      /* b1 */ {bool_int8_t_##funcnameT, bool_int16_t_##funcnameT, bool_int_##funcnameT, bool_int64_t_##funcnameT, bool_float_##funcnameT, bool_double_##funcnameT, bool_bool_##funcnameT}, \
      } \
};

#define DEFINE_FUNC_ARRAY_TEMPLATES(funcname, ...) \
typedef void (*funcname##_type)(__VA_ARGS__); \
static constexpr std:: \
    array<std::array<funcname##_type, kDType2Index.size()>, kDType2Index.size()> \
      kFuncsLookup = {{ \
      /*        ki1  ki2  ki4  ki8  kf4  kf8  kb1*/ \
      /* i1 */  {funcname<int8_t,int8_t>,funcname<int8_t,int16_t>,funcname<int8_t,int>,funcname<int8_t,int64_t>,funcname<int8_t,float>,funcname<int8_t,double>,funcname<int8_t,bool>}, \
                {funcname<int16_t,int8_t>,funcname<int16_t,int16_t>,funcname<int16_t,int>,funcname<int16_t,int64_t>,funcname<int16_t,float>,funcname<int16_t,double>,funcname<int16_t,bool>}, \
                {funcname<int,int8_t>,funcname<int,int16_t>,funcname<int,int>,funcname<int,int64_t>,funcname<int,float>,funcname<int,double>,funcname<int,bool>}, \
                {funcname<int64_t,int8_t>,funcname<int64_t,int16_t>,funcname<int64_t,int>,funcname<int64_t,int64_t>,funcname<int64_t,float>,funcname<int64_t,double>,funcname<int64_t,bool>}, \
                {funcname<float,int8_t>,funcname<float,int16_t>,funcname<float,int>,funcname<float,int64_t>,funcname<float,float>,funcname<float,double>,funcname<float,bool>}, \
                {funcname<double,int8_t>,funcname<double,int16_t>,funcname<double,int>,funcname<double,int64_t>,funcname<double,float>,funcname<double,double>,funcname<double,bool>}, \
                {funcname<bool,int8_t>,funcname<bool,int16_t>,funcname<bool,int>,funcname<bool,int64_t>,funcname<bool,float>,funcname<bool,double>,funcname<bool,bool>} \
      } \
};

}