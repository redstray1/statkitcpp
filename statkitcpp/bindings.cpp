#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";
}