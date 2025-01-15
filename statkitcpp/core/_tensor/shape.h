#ifndef SHAPE_HEADER_H
#define SHAPE_HEADER_H

#include <vector>
#include <cstdint>
#include <string>

bool IsBroadcastable(const std::vector<uint32_t>& shape1,
                     const std::vector<uint32_t>& shape2);

std::vector<uint32_t> BroadcastShapes(const std::vector<uint32_t>& shape1,
                                      const std::vector<uint32_t>& shape2);

std::string ShapeToString(const std::vector<uint32_t>& shape);

std::vector<uint32_t> RemoveDim(const std::vector<uint32_t>& shape, int dim = -1, bool keepdims = false);
#endif