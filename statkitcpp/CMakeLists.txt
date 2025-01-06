add_library(libcore STATIC
    core/tensor.cpp
    core/autograd.cpp
)
set_target_properties(libcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_include_directories(statkitcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

pybind11_add_module(_statkitcpp bindings.cpp)
target_link_libraries(_statkitcpp PRIVATE libcore)

install(TARGETS _statkitcpp DESTINATION .)