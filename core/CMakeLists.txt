cmake_minimum_required(VERSION 3.0)
project(Minitorch)

# use ascend device
option(USE_ASCEND "Enable Ascend device support" ON)

if(USE_ASCEND)
    # include(ExternalProject)
    # # 定义 Ascend 库的路径
    # set(ASCEND_DIR "${CMAKE_SOURCE_DIR}/ascend")
    # set(ASCEND_BUILD_SCRIPT "${ASCEND_DIR}/build.sh")
    # set(ASCEND_BUILD_DIR "${CMAKE_BINARY_DIR}/ascend_build")
    # # 添加 ExternalProject 以构建 Ascend 库
    # ExternalProject_Add(
    # ascend
    # PREFIX ${ASCEND_BUILD_DIR}
    # SOURCE_DIR ${ASCEND_DIR}
    # CONFIGURE_COMMAND ""  # No configure command
    # BUILD_COMMAND ${ASCEND_BUILD_SCRIPT}  # Run the build script
    # INSTALL_COMMAND ""  # No install command
    # )

    # # 添加 Ascend 的 include 目录到目标
    # ExternalProject_Get_Property(ascend install_dir)
    # set(ASCEND_INCLUDE_PATH "${ASCEND_DIR}/include")
    # target_include_directories(my_executable PRIVATE ${ASCEND_INCLUDE_PATH})
    # # 使 `my_executable` 依赖于 Ascend 项目的编译
    # add_dependencies(my_executable ascend)
    set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
    set(SOC_VERSION "Ascend310b4" CACHE STRING "system on chip type")
    message(STATUS "Ascend device support enabled.")
    set(ASCEND_LIBRARY_HOME "${CMAKE_CURRENT_SOURCE_DIR}/../ascend/out" CACHE STRING "ASCEND library path")
    message(STATUS "ASCEND library home: ${ASCEND_LIBRARY_HOME}")
    set(ASCEND_INCLUDE_PATH "${ASCEND_LIBRARY_HOME}/include/ascend_kernels_${RUN_MODE}" CACHE STRING "ASCEND include path")
    message(STATUS "ASCEND include path: ${AsCEND_INCLUDE_PATH}")
    set(CANN_INCLUDE_PATH "/opt/ascend/ascend-toolkit/latest/include" CACHE STRING "CANN include path")
    set(ASCEND_LIBRARY_PATH "${ASCEND_LIBRARY_HOME}/lib" CACHE STRING "ASCEND library path")
    message(STATUS "ASCEND library path: ${ASCEND_LIBRARY_PATH}")

    add_subdirectory(src/ascend/)

else()
    message(STATUS "Ascend device support disabled.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Set src and include directories
set(MINITORCH_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MINITORCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set build directory
set(MINITORCH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)

# Set install directory
set(MINITORCH_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)

# set ascend kernal source files
file(GLOB_RECURSE ASCEND_KERNAL_SRC_FILES
    ${MINITORCH_SRC_DIR}/ascend/*.cpp
    ${MINITORCH_SRC_DIR}/ascend/*.h
)
message(STATUS "Ascend kernal source files: ${ASCEND_KERNAL_SRC_FILES}")

# set_source_files_properties(${ASCEND_KERNAL_SRC_FILES} PROPERTIES LANGUAGE CCE)
set(ASCEND_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ascend_device_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ascend_device_api.h)

# set src files
file(GLOB_RECURSE MINITORCH_SRC_FILES
    ${MINITORCH_SRC_DIR}/*.cpp
    ${MINITORCH_SRC_DIR}/*.h)

# exclude ascend source files
if(USE_ASCEND)
    foreach(file ${ASCEND_KERNAL_SRC_FILES})
        list(REMOVE_ITEM MINITORCH_SRC_FILES ${file})
    endforeach()
else()
    foreach(file ${ASCEND_FILES})
        list(REMOVE_ITEM MINITORCH_SRC_FILES ${file})
    endforeach()
endif()

message(STATUS "Source files: ${MINITORCH_SRC_FILES}")

# # include ascend cmake
# add_subdirectory(src/ascend/)
# link_directories(/home/lqb/python/minitorch/ascend/out/lib
#                 /opt/ascend/ascend-toolkit/latest/lib64)

# build shared library
add_library(minitorch SHARED ${MINITORCH_SRC_FILES})
target_include_directories(minitorch PUBLIC ${MINITORCH_INCLUDE_DIR})

if(USE_ASCEND)
    target_include_directories(minitorch PUBLIC 
        # ${ASCEND_INCLUDE_PATH}
        ${CANN_INCLUDE_PATH}
    )

    target_link_directories(minitorch PRIVATE
        /opt/ascend/ascend-toolkit/latest/lib64
    )

    target_link_libraries(minitorch PUBLIC
        # $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::${SOC_VERSION}>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:c_sec>>
        ascend_kernels_${RUN_MODE}
        tiling_api
        register
        platform
        ascendalog
        dl
    )
endif()

target_compile_options(minitorch PUBLIC 
    -Wall # 启用所有警告
    -Wextra # 启用额外警告
    -O1 # 优化级别 2
    -std=c++17 # C++ 17 标准
    -g
)

# build test
add_executable(tensor_test ${CMAKE_CURRENT_SOURCE_DIR}/test/tensor_test.cpp)

# link minitorch library
target_link_libraries(tensor_test minitorch)

# set target compile options
target_compile_options(tensor_test PUBLIC 
    -Wall # 启用所有警告
    -Wextra # 启用额外警告
    -O1 # 优化级别 2
    -std=c++17 # C++ 17 标准
    -g
)