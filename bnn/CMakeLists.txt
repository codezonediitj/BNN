add_library(bnn_utils SHARED
            utils/utils_impl.cpp)
add_library(bnn_core SHARED
            core/tensor_impl.cpp
            utils/utils_impl.cpp)
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LINK_LIBRARIES gtest gtest_main pthread)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core ${LINK_LIBRARIES} bnn_core)
endif()
