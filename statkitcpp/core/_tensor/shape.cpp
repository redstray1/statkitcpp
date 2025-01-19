#include "shape.h"
#include "../errors.h"
#include <vector>

bool CheckIndex(int index, size_t size) {
    return 0 <= index && index < static_cast<int>(size);
}

bool IsBroadcastable(const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2) {
    for (int i = 0; i < std::max(shape1.size(), shape2.size()); i++) {
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
    for (int i = 0; i < std::max(shape1.size(), shape2.size()); i++) {
        int index1 = static_cast<int>(shape1.size()) - i - 1;
        int index2 = static_cast<int>(shape2.size()) - i - 1;
        size_t dim1 = CheckIndex(index1, shape1.size()) ? shape1[index1] : 1;
        size_t dim2 = CheckIndex(index2, shape2.size()) ? shape2[index2] : 1;
        output_shape[output_shape.size() - i - 1] = std::max(dim1, dim2);
    }
    return output_shape;
}


std::vector<size_t> RemoveDim(const std::vector<size_t>& shape, int dim, bool keepdims) {
    std::vector<size_t> new_shape(shape.size() - (1 - keepdims));
    if (dim < 0) {
        dim += shape.size();
    }
    if (dim < 0 || dim >= shape.size()) {
        throw DimError{dim, shape.size()};
    }
    for (size_t i = 0; i < dim; i++) {
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