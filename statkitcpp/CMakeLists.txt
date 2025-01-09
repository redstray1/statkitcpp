add_library(libcore STATIC
    core/_tensor/tensor.cpp
    core/autograd.cpp
)
set_target_properties(libcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/_tensor)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/dispatcher)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

pybind11_add_module(_statkitcpp bindings.cpp)
target_link_libraries(_statkitcpp PRIVATE libcore)

install(TARGETS _statkitcpp DESTINATION .)