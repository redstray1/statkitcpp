#ifndef SHAPE_HEADER_H
#define SHAPE_HEADER_H

#include <vector>
#include <string>

bool IsBroadcastable(const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2);

std::vector<size_t> BroadcastShapes(const std::vector<size_t>& shape1,
                                      const std::vector<size_t>& shape2);

std::string ShapeToString(const std::vector<size_t>& shape);

std::vector<size_t> RemoveDim(const std::vector<size_t>& shape, int dim = -1, bool keepdims = false);
#endif