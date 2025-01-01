file(GLOB SRC_TEST CONFIGURE_DEPENDS "tensors/*.cpp")
add_catch(test_tensor ${SRC_TEST})
target_link_libraries(test_tensor PRIVATE libcore)