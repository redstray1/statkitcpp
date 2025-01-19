#include "Scalar.h"
#include "ScalarType.h"
#include "errors.h"

namespace statkitcpp {

void SKPP_CHECK(const Scalar& arg, ScalarType type) { //NOLINT
    if (!CanCast(arg.Type(), type)) {
        throw TypeCastError{GetDTypeName(arg.Type()), GetDTypeName(type)};
    }
}

}