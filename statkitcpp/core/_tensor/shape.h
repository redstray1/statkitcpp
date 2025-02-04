#ifndef SHAPE_HEADER_H
#define SHAPE_HEADER_H

#include <vector>
#include "DimMask.h"
#include <string>

bool IsBroadcastable(const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2);

bool IsEqualSuffix(const std::vector<size_t>& src,
                   const std::vector<size_t>& to_check);

std::vector<size_t> BroadcastShapes(const std::vector<size_t>& shape1,
                                    const std::vector<size_t>& shape2);

std::vector<size_t> BroadcastShapesForMatMul(const std::vector<size_t>& shape1,
                                    const std::vector<size_t>& shape2, bool rhs_transposed = false);

std::string ShapeToString(const std::vector<size_t>& shape);

std::vector<size_t> RemoveDim(const std::vector<size_t>& shape, int dim = -1, bool keepdims = false);

size_t GetFlatIndex(const std::vector<size_t>& indexes, 
                    const std::vector<size_t>& strides,
                    const std::vector<size_t>& shape,
                    size_t itemsize);

std::vector<size_t> GetMultiIndex(const size_t& flat_index,
                                  const std::vector<size_t>& strides,
                                  size_t itemsize,
                                  size_t size);

DimMask GetMaskOfOnes(const std::vector<size_t>& shape);

#endif