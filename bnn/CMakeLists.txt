add_library(bnn_utils SHARED
            utils/utils_impl.cpp)
add_library(bnn_core SHARED
            core/tensor_impl.cpp)
target_link_libraries(bnn_core bnn_utils)
if(BUILD_CUDA)
    add_library(bnn_cuda_utils SHARED
                cuda/utils/cuda_wrappers_impl.cu)
    add_library(bnn_cuda_core SHARED
                cuda/core/tensor_impl.cu)
    target_link_libraries(bnn_cuda_core bnn_core bnn_cuda_utils)
endif()
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LINK_LIBRARIES gtest gtest_main pthread)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core ${LINK_LIBRARIES} bnn_core)
    if(BUILD_CUDA)
        add_executable(test_cuda_core tests/test_cuda_core.cpp)
        target_link_libraries(test_cuda_core ${LINK_LIBRARIES}
                              bnn_cuda_core)
    endif()
endif()
