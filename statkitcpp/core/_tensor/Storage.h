#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <utility>
#include <stdlib.h>
#include "ScalarType.h"
#include "errors.h"

namespace statkitcpp {

class Storage {
private:
    void* data_ptr_;
    size_t size_bytes_;
public:
    Storage(size_t n_bytes) { data_ptr_ = malloc(n_bytes); size_bytes_ = n_bytes; }
    Storage(void* data_ptr, size_t n_bytes) : data_ptr_(data_ptr), size_bytes_(n_bytes) {}
    Storage& operator=(const Storage& other) {
        if (this == &other) {
            return *this;
        }
        size_bytes_ = other.size_bytes_;
        data_ptr_ = malloc(size_bytes_);
        memcpy(data_ptr_, other.data_ptr_, size_bytes_);
        return *this;
    }
    Storage& operator=(Storage&& other) {
        if (this == &other) {
            return *this;
        }
        size_bytes_ = std::move(other.size_bytes_);
        ClearData();
        data_ptr_ = other.data_ptr_;

        other.data_ptr_ = nullptr;
        return *this;
    }
    Storage() { data_ptr_ = nullptr; size_bytes_ = 0; };
    Storage(Storage&& other) {
        if (this == &other) {
            return;
        }
        size_bytes_ = std::move(other.size_bytes_);
        ClearData();
        data_ptr_ = other.data_ptr_;
    }
    Storage(const Storage& other) {
        if (this == &other) {
            return;
        }
        size_bytes_ = other.size_bytes_;
        data_ptr_ = malloc(size_bytes_);
        memcpy(data_ptr_, other.data_ptr_, size_bytes_);
    }
    ~Storage() {
        ClearData();
    }

    void ClearData() {
        if (data_ptr_ != nullptr) {
            free(data_ptr_);
            data_ptr_ = nullptr;
        }
    }

    void Reset() {
        ClearData();
        size_bytes_ = 0;
    }

    size_t GetNbytes() const {
        return size_bytes_;
    }

    const void* GetDataPtr() const {
        return data_ptr_;
    }

    void* GetDataPtr() {
        return data_ptr_;
    }

    template <typename T>
    T GetElement(size_t offset) const {
        return *(static_cast<T*>(data_ptr_) + offset);
    }

    std::string ToString(size_t offset, ScalarType dtype) const {
        std::stringstream ss;
        if (size_bytes_ % ItemSize(dtype) != 0) {
            throw InvalidDatatypeError{};
        }
        #define DEFINE_REPR(T, name) \
        case ScalarType::name:{ \
            T value = *(static_cast<T*>(data_ptr_) + offset); \
            if constexpr(std::is_same_v<T, int8_t>) { \
                ss << static_cast<int16_t>(value); \
            } else { \
                ss << value; \
            } \
            return ss.str(); \
        }
        
        switch(dtype) {
            SCALAR_TYPES(DEFINE_REPR)
            default:
                return "Nan";
        }
        #undef DEFINE_REPR
    }

    operator bool() const {
        return data_ptr_;
    }

};

}