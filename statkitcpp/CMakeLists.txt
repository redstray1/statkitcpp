add_library(_statcore STATIC
    core/tensor.cpp,
    core/autograd.cpp,
)

target_include_directories(_statcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

find_package(pybind11 REQUIRED)

pybind11_add_module(_statcore_bindings bindings.cpp)

target_link_libraries(_statcore_bindings PRIVATE _statcore)