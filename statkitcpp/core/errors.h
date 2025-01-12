#ifndef ERRORS_HEADER_H
#define ERRORS_HEADER_H
#include <cstdint>
#include <stdexcept>
#include <string>
class SliceError : public std::invalid_argument {
public:
    explicit SliceError(const std::string& reason)
        : std::invalid_argument{"Wrong slice argument: " + reason} {
    }
};
class InvalidDatatypeError : public std::runtime_error {
public:
    explicit InvalidDatatypeError()
        : std::runtime_error{"Unsupported dtype. Supported: float32, float64"} {
    }
};
class BroadcastError : public std::invalid_argument {
public:
    explicit BroadcastError(const std::string& shape1, const std::string& shape2)
        : std::invalid_argument{"Shapes " + shape1 + " and " + shape2 + " are not compatible for broadcasting."}{
    }
};
class OutOfRangeFlatError : public std::out_of_range {
public:
    explicit OutOfRangeFlatError(uint32_t flat_index)
        : std::out_of_range{"Flat index " + std::to_string(flat_index) + " out of bounds for the given shape."}{
    }
};
class OutOfRangeError : public std::out_of_range {
public:
    explicit OutOfRangeError(uint32_t dim, uint32_t dim_index)
        : std::out_of_range{"Index " + std::to_string(dim_index) + " out of bounds for the given dimension " + std::to_string(dim)}{
    }
};
#endif