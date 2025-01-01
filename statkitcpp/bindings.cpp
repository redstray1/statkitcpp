#include <pybind11/pybind11.h>
#include "core/config.h"

namespace py = pybind11;

PYBIND11_MODULE(statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";
    py::enum_<data_type>(m, "DataType")
        .value("Float8", data_type::Float8)
        .value("Float16", data_type::Float16)
        .value("Float32", data_type::Float32)
        .value("Float64", data_type::Float64);
}