#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Slice.h"


namespace py = pybind11;

std::optional<int> ParseInt(py::object obj) {
    if (py::isinstance<py::none>(obj)) {
        return std::nullopt;
    } else if (py::isinstance<py::int_>(obj)) {
        return obj.cast<int>();
    } else {
        throw std::invalid_argument{"Argument is not an int"};
    }
}

statkitcpp::Slice ParseSlice(py::slice pyslice) {
    auto start = ParseInt(py::getattr(pyslice, "start"));
    auto end = ParseInt(py::getattr(pyslice, "stop"));
    auto step = ParseInt(py::getattr(pyslice, "step"));
    return statkitcpp::Slice(start.has_value() ? start.value() : 0, end.value(), step.has_value() ? step.value() : 1);
}