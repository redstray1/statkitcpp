#pragma once

#include <sstream>
#include "../core/_tensor/ScalarType.h"

namespace statkitcpp {

std::string ToString(void* data, ScalarType dtype) {
    std::stringstream ss;
    #define DEFINE_REPR(_, name) \
    case ScalarType::name: \
        ss << *(static_cast<ScalarTypeToCPPTypeT<ScalarType::name>*>(data)); \
        return ss.str();
    
    switch(dtype) {
        SCALAR_TYPES(DEFINE_REPR)
        default:
            return "Nan";
    }
    #undef DEFINE_REPR
}

}