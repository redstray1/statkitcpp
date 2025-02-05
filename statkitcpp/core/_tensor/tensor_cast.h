#include <cstddef>
#include "ScalarType.h"
#include "promote_type.h"
#include "UFunc.h"

namespace statkitcpp {

namespace ops {

template <typename From, typename To>
void copy(void* src, void* dest, size_t outsize) { //NOLINT
    From* src_t = static_cast<From*>(src);
    To* dest_t = static_cast<To*>(dest);
    for (size_t i = 0; i < outsize; i++, src_t++, dest_t++) {
        *dest_t = static_cast<To>(*src_t);
    }
}

void copy(void* src, ScalarType dtype1, void* dest, ScalarType dtype2, size_t outsize) { //NOLINT
    DEFINE_FUNC_ARRAY_TEMPLATES(copy, void*, void*, size_t);
    auto idx1 = kDType2Index[static_cast<int64_t>(dtype1)];
    auto idx2 = kDType2Index[static_cast<int64_t>(dtype2)];
    kFuncsLookup[idx1][idx2](src, dest, outsize);
}

}

}