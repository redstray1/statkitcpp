#include "Storage.h"

namespace statkitcpp {
    bool IsSharedStorage(const Storage& storage0, const Storage& storage1) {
        return storage0.GetDataPtr() == storage1.GetDataPtr();
    }
}