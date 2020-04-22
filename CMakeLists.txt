cmake_minimum_required(VERSION 3.10.2)
project(adaboost CXX)

# First, define all the compilation options.
option(BUILD_TESTS "Build tests." OFF)
option(INSTALL_GOOGLETEST "For installing GoogleTest on your system along with the build." OFF)
option(BUILD_CUDA "Build with CUDA support." OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "Building without CUDA support as no CUDA compiler is found on this system.")
endif()


# Set required standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Install GoogleTest
if(INSTALL_GOOGLETEST AND BUILD_TESTS)
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG master
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest-src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest-build"
    TEST_COMMAND "")
endif()

# Set include directories
include_directories(.)

# Recurse into rest of the project
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/libs)
link_directories(${CMAKE_BINARY_DIR}/libs)
add_subdirectory(bnn)

install(DIRECTORY bnn DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN "*.cpp" EXCLUDE
        PATTERN "tests" EXCLUDE
        )
install(FILES
        ${CMAKE_BINARY_DIR}/libs/libbnn_core.so
        ${CMAKE_BINARY_DIR}/libs/libbnn_utils.so
        ${CMAKE_BINARY_DIR}/libs/libbnn_operations.so
        ${CMAKE_BINARY_DIR}/libs/libbnn_autodiff.so
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_CUDA)
  install(FILES
          ${CMAKE_BINARY_DIR}/libs/libbnn_cuda.so
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
