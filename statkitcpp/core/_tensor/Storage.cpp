#include "Storage.h"
#include "ScalarType.h"

namespace statkitcpp {
    namespace ops {
        template <typename To, typename From>
        void storage_copy(To* dest, From* src) { //NOLINT

        }
    }
    bool IsSharedStorage(const Storage& storage0, const Storage& storage1) {
        return storage0.GetDataPtr() == storage1.GetDataPtr();
    }


    // void StorageCopy(Storage& dest, const ScalarType& dest_type, const Storage& src, ScalarType src_type) {
        
    // }

}