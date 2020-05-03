add_library(bnn_utils SHARED
            utils/utils_impl.cpp)
add_library(bnn_core SHARED
            core/tensor_impl.cpp
            core/tensor_ops_impl.cpp)
add_library(bnn_operations SHARED
            operations/operations_impl.cpp
            operations/operators_impl.cpp)
add_library(bnn_autodiff SHARED
            autodiff/graph_impl.cpp
            autodiff/forward_impl.cpp
            autodiff/reverse_impl.cpp)
add_library(bnn_io SHARED
            io/input_impl.cpp)
target_link_libraries(bnn_utils pthread)
target_link_libraries(bnn_core bnn_utils m)
target_link_libraries(bnn_operations bnn_utils)
target_link_libraries(bnn_autodiff bnn_utils)
target_link_libraries(bnn_io bnn_utils)
if(BUILD_CUDA)
    add_library(bnn_cuda_utils SHARED
                cuda/utils/cuda_wrappers_impl.cu)
    add_library(bnn_cuda_core SHARED
                cuda/core/tensor_impl.cu)
    target_link_libraries(bnn_cuda_core bnn_core bnn_cuda_utils)
endif()
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(GTEST gtest gtest_main pthread)
    add_executable(test_core tests/test_core.cpp)
    set(CORE bnn_core)
    target_link_libraries(test_core ${GTEST} ${CORE})
    add_executable(test_operations tests/test_operations.cpp)
    set(OPERATIONS bnn_operations bnn_core)
    target_link_libraries(test_operations ${GTEST} ${OPERATIONS})
    add_executable(test_autodiff tests/test_autodiff.cpp)
    set(AUTODIFF bnn_autodiff bnn_operations bnn_core)
    target_link_libraries(test_autodiff ${GTEST} ${AUTODIFF})
    if(BUILD_CUDA)
        add_executable(test_cuda_core tests/test_cuda_core.cpp)
        set(CUDA_CORE bnn_cuda_core)
        target_link_libraries(test_cuda_core ${GTEST} ${CUDA_CORE})
    endif()
endif()
