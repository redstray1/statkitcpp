add_library(libcore STATIC
    core/tensor.cpp
    core/autograd.cpp
)

target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

find_package(pybind11 REQUIRED)

pybind11_add_module(libcore_bindings bindings.cpp)

target_link_libraries(libcore_bindings PRIVATE libcore)