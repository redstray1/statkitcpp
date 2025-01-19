#ifndef ERRORS_HEADER_H
#define ERRORS_HEADER_H
#include <stdexcept>
#include <string>
class SliceError : public std::invalid_argument {
public:
    explicit SliceError(const std::string& reason)
        : std::invalid_argument{"Wrong slice argument: " + reason} {
    }
};
class ReshapeError : public std::runtime_error {
public:
    explicit ReshapeError(const std::string& size, const std::string& shape)
        : std::runtime_error{"Cannot reshape tensor of size " + size + " into shape " + shape} {
    }
};
class InvalidDatatypeError : public std::runtime_error {
public:
    explicit InvalidDatatypeError()
        : std::runtime_error{"Unsupported dtype. Supported: int8, int16, int32, int64, float32, float64, bool"} {
    }
};
class TypeCastError : public std::runtime_error {
public:
    explicit TypeCastError(const std::string& type1, const std::string& type2)
        : std::runtime_error{"Can not cast " + type1 + " to " + type2} {
    }
};
class BroadcastError : public std::invalid_argument {
public:
    explicit BroadcastError(const std::string& shape1, const std::string& shape2)
        : std::invalid_argument{"Shapes " + shape1 + " and " + shape2 + " are not compatible for broadcasting."}{
    }
};
class DimError : public std::out_of_range {
public:
    explicit DimError(int dim, size_t dims)
        : std::out_of_range{"Axis " + std::to_string(dim) + " out of bound of tensor of dimension " + std::to_string(dims)} {
    }
};
class OutOfRangeFlatError : public std::out_of_range {
public:
    explicit OutOfRangeFlatError(size_t flat_index, size_t size)
        : std::out_of_range{"Flat index " + std::to_string(flat_index) + " out of bounds for the array of size " + std::to_string(size)}{
    }
};
class OutOfRangeError : public std::out_of_range {
public:
    explicit OutOfRangeError(size_t dim, size_t dim_index)
        : std::out_of_range{"Index " + std::to_string(dim_index) + " out of bounds for the given dimension " + std::to_string(dim)}{
    }
};
#endif