#include "shape.h"
#include "../errors.h"
#include <cstdint>

bool CheckIndex(int index, uint32_t size) {
    return 0 <= index && index < static_cast<int>(size);
}

bool IsBroadcastable(const std::vector<uint32_t>& shape1,
                     const std::vector<uint32_t>& shape2) {
    for (int i = 0; i < std::max(shape1.size(), shape2.size()); i++) {
        int index1 = static_cast<int>(shape1.size()) - i - 1;
        int index2 = static_cast<int>(shape2.size()) - i - 1;
        uint32_t dim1 = CheckIndex(index1, shape1.size()) ? shape1[index1] : 1;
        uint32_t dim2 = CheckIndex(index2, shape2.size()) ? shape2[index2] : 1;
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

std::string ShapeToString(const std::vector<uint32_t>& shape) {
    std::string shape_repr = "(";
    for (uint32_t i = 0; i < shape.size(); i++) {
        const auto dim = shape[i];
        shape_repr += std::to_string(dim);
        if (i < shape.size() - 1) {
            shape_repr += ',';
        }
    }
    shape_repr += ')';
    return shape_repr;
}

std::vector<uint32_t> BroadcastShapes(const std::vector<uint32_t>& shape1,
                                      const std::vector<uint32_t>& shape2) {
    if (!IsBroadcastable(shape1, shape2)) {
        throw BroadcastError{ShapeToString(shape1), ShapeToString(shape2)};
    }
    std::vector<uint32_t> output_shape(std::max(shape1.size(), shape2.size()));
    for (int i = std::max(shape1.size(), shape2.size()) - 1; i >= 0; i--) {
        uint32_t lhs_dim = i < static_cast<int>(shape1.size()) ? shape1[i] : 1;
        uint32_t rhs_dim = i < static_cast<int>(shape2.size()) ? shape2[i] : 1;
        output_shape[i] = std::max(lhs_dim, rhs_dim);
    }
    return output_shape;
}