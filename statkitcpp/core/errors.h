#ifndef ERRORS_HEADER_H
#define ERRORS_HEADER_H
#include <stdexcept>
#include <string>
#include <format>
class SliceError : public std::invalid_argument {
public:
    explicit SliceError(const std::string& reason)
        : std::invalid_argument{"Wrong slice argument: " + reason} {
    }
};
class ReshapeError : public std::runtime_error {
public:
    explicit ReshapeError(size_t size, const std::string& shape)
        : std::runtime_error{"Cannot reshape tensor of size " + std::to_string(size) + " into shape " + shape} {
    }
};
class DotOperationError : public std::runtime_error {
public:
    explicit DotOperationError(size_t size1, size_t size2)
        : std::runtime_error{"Cannot calculate dot product of vectors with sizes " + std::to_string(size1) + " and " + std::to_string(size2)} {
    }
};
class MatMulError : public std::runtime_error {
public:
    explicit MatMulError(size_t n, size_t m)
        : std::runtime_error{std::format("MatMul: Input operand 1 has a mismatch in its core dimension 0, with signature (n?,k),(k,m?)->(n?,m?) (size {} is different from {})", n, m)} {
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
class DegreesOfFreedomError : public std::runtime_error {
public:
    explicit DegreesOfFreedomError()
        : std::runtime_error{"var(): degrees of freedom is <= 0."} {
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
class InputsAndGradMismatchError : public std::runtime_error {
public:
    explicit InputsAndGradMismatchError(size_t input_size, size_t grad_size)
        : std::runtime_error{"Mismatch number of inputs=" + std::to_string(input_size) + " and number of resulting gradients=" + std::to_string(grad_size) + " during backward."}{
    }
};
#endif