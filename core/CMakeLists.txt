cmake_minimum_required(VERSION 3.0)
project(Minitorch)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Set src and include directories
set(MINITORCH_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MINITORCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
# Set build directory
set(MINITORCH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)
# Set install directory
set(MINITORCH_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)

# set src files
file(GLOB_RECURSE MINITORCH_SRC_FILES 
    ${MINITORCH_SRC_DIR}/*.cpp
    ${MINITORCH_SRC_DIR}/*.h)
message(STATUS "Source files: ${MINITORCH_SRC_FILES}")

# build shared library
add_library(minitorch SHARED ${MINITORCH_SRC_FILES})
target_include_directories(minitorch PUBLIC ${MINITORCH_INCLUDE_DIR})
target_compile_options(minitorch PRIVATE 
    -Wall              # 启用所有警告
    -Wextra            # 启用额外警告
    -O1                # 优化级别 2
    -std=c++17         # C++ 17 标准
    -g
)

# build test
add_executable(tensor_test ${CMAKE_CURRENT_SOURCE_DIR}/test/tensor_test.cpp)
# link minitorch library
target_link_libraries(tensor_test minitorch)
# set target compile options
target_compile_options(tensor_test PRIVATE 
    -Wall              # 启用所有警告
    -Wextra            # 启用额外警告
    -O1                # 优化级别 2
    -std=c++17         # C++ 17 标准
    -g
)