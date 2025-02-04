#include "shape.h"
#include "../errors.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cassert>

bool CheckIndex(int index, size_t size) {
    return 0 <= index && index < static_cast<int>(size);
}

bool IsBroadcastable(const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2) {
    for (int i = 0; i < static_cast<int>(std::max(shape1.size(), shape2.size())); i++) {
        int index1 = static_cast<int>(shape1.size()) - i - 1;
        int index2 = static_cast<int>(shape2.size()) - i - 1;
        size_t dim1 = CheckIndex(index1, shape1.size()) ? shape1[index1] : 1;
        size_t dim2 = CheckIndex(index2, shape2.size()) ? shape2[index2] : 1;
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

bool IsEqualSuffix(const std::vector<size_t>& src,
                   const std::vector<size_t>& to_check) {
    if (to_check.size() > src.size()) { return false; }
    for (int i = src.size() - 1, j = to_check.size() - 1; j >= 0; j--, i--) {
        if (src[i] != to_check[j]) {
            return false;
        }
    }
    return true;
}

std::string ShapeToString(const std::vector<size_t>& shape) {
    std::string shape_repr = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = shape[i];
        shape_repr += std::to_string(dim);
        if (i < shape.size() - 1) {
            shape_repr += ',';
        }
    }
    shape_repr += ')';
    return shape_repr;
}

std::vector<size_t> BroadcastShapes(const std::vector<size_t>& shape1,
                                      const std::vector<size_t>& shape2) {
    if (!IsBroadcastable(shape1, shape2)) {
        throw BroadcastError{ShapeToString(shape1), ShapeToString(shape2)};
    }
    std::vector<size_t> output_shape(std::max(shape1.size(), shape2.size()));
    for (int i = 0; i < static_cast<int>(std::max(shape1.size(), shape2.size())); i++) {
        int index1 = static_cast<int>(shape1.size()) - i - 1;
        int index2 = static_cast<int>(shape2.size()) - i - 1;
        size_t dim1 = CheckIndex(index1, shape1.size()) ? shape1[index1] : 1;
        size_t dim2 = CheckIndex(index2, shape2.size()) ? shape2[index2] : 1;
        output_shape[output_shape.size() - i - 1] = std::max(dim1, dim2);
    }
    return output_shape;
}

std::vector<size_t> BroadcastShapesForMatMul(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2, bool rhs_transposed) {
    if (shape1.size() <= 2 && shape2.size() <= 2) {
        throw std::runtime_error{"Wrong arguments in BroadcastShapesForMatMul"};
    }
    if (shape1.size() == 1) {
        if (shape1[0] != shape2[shape2.size() - 2 + rhs_transposed]) {
            throw MatMulError{shape1[0], shape2[shape2.size() - 2 + rhs_transposed]};
        }
        std::vector<size_t> output_shape(shape2);
        if (!rhs_transposed) {
            std::swap(output_shape[output_shape.size() - 1], output_shape[output_shape.size() - 2]);
        }
        output_shape.pop_back();
        return output_shape;
    }
    if (shape2.size() == 1) {
        if (shape1.back() != shape2[0]) {
            throw MatMulError{shape1.back(), shape2[0]};
        }
        std::vector<size_t> output_shape(shape1);
        output_shape.pop_back();
        return output_shape;
    }
    if (shape1.size() == 2) {
        if (shape1.back() != shape2[shape2.size() - 2 + rhs_transposed]) {
            throw MatMulError{shape1.back(), shape2[shape2.size() - 2 + rhs_transposed]};
        }
        std::vector<size_t> output_shape(shape2);
        if (rhs_transposed) {
            std::swap(output_shape.back(), output_shape[output_shape.size() - 2]);
        }
        output_shape[output_shape.size() - 2] = shape1[0];
        return output_shape;
    }
    if (shape2.size() == 2) {
        if (shape1.back() != shape2[rhs_transposed]) {
            throw MatMulError{shape1.back(), shape2[rhs_transposed]};
        }
        std::vector<size_t> output_shape(shape1);
        output_shape.back() = shape2[1 - rhs_transposed];
        return output_shape;
    }
    std::vector<size_t> output_shape(std::max(shape1.size(), shape2.size()));
    for (int i = 0; i < static_cast<int>(output_shape.size() - 2); i++) {
        int index1 = static_cast<int>(shape1.size()) - i - 3;
        int index2 = static_cast<int>(shape2.size()) - i - 3;
        size_t dim1 = CheckIndex(index1, shape1.size()) ? shape1[index1] : 1;
        size_t dim2 = CheckIndex(index2, shape2.size()) ? shape2[index2] : 1;
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw BroadcastError{ShapeToString(shape1), ShapeToString(shape2)};
        }
        output_shape[output_shape.size() - i - 3] = std::max(dim1, dim2);
    }
    output_shape[output_shape.size() - 2] = shape1[shape1.size() - 2];
    output_shape.back() = shape2[shape2.size() - 1 - rhs_transposed];
    return output_shape;
}

std::vector<size_t> RemoveDim(const std::vector<size_t>& shape, int dim, bool keepdims) {
    std::vector<size_t> new_shape(shape.size() - (1 - keepdims));
    if (dim < 0) {
        dim += shape.size();
    }
    if (dim < 0 || dim >= static_cast<int>(shape.size())) {
        throw DimError{dim, shape.size()};
    }
    for (size_t i = 0; i < static_cast<size_t>(dim); i++) {
        new_shape[i] = shape[i];
    }
    for (size_t i = dim + 1; i < shape.size(); i++) {
        new_shape[i - (1 - keepdims)] = shape[i];
    }
    if (keepdims) {
        new_shape[dim] = 1;
    }
    return new_shape;
}

size_t GetFlatIndex(const std::vector<size_t>& indexes, 
                    const std::vector<size_t>& strides,
                    const std::vector<size_t>& shape,
                    [[maybe_unused]]size_t itemsize) {
    assert(indexes.size() <= shape.size());
    size_t index = 0;
    for (size_t i = 0; i < indexes.size(); i++) {
        if (indexes[i] >= shape[i]) {
            throw OutOfRangeError{i, indexes[i]};
        }
        index += indexes[i] * strides[i];
    }
    return index;
}

std::vector<size_t> GetMultiIndex(const size_t& flat_index,
                                  const std::vector<size_t>& strides,
                                  [[maybe_unused]]size_t itemsize,
                                  size_t size) {
    std::vector<size_t> indexes(strides.size(), 0);
    if (flat_index >= size) {
        throw OutOfRangeFlatError{flat_index, size};
    }
    size_t index = flat_index;
    for (size_t i = 0; i < strides.size(); ++i) {
        indexes[i] = index / strides[i];
        index %= strides[i];
    }
    return indexes;
}

DimMask GetMaskOfOnes(const std::vector<size_t>& shape) {
    DimMask mask(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 1) {
            mask.Set(i);
        }
    }
    return mask;
}