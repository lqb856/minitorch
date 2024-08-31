cmake_minimum_required(VERSION 3.0)
project(Minitorch)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# use ascend device
option(USE_ASCEND "Enable Ascend device support" ON)
option(CMAKE_BUILD_TYPE "Build type Release/Debug (default Debug)" Debug)

if(USE_ASCEND)
    # optional parameters
    set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
    set(SOC_VERSION "Ascend310b4" CACHE STRING "system on chip type")
    set(ASCEND_CANN_PACKAGE_PATH "/opt/ascend/ascend-toolkit/latest" CACHE STRING "ASCEND CANN package installation directory")
    set(ASCEND_CANN_INCLUDE_PATH "${ASCEND_CANN_PACKAGE_PATH}/include" CACHE STRING "CANN include path")
    set(ASCEND_CANN_LIBRARY_PATH "${ASCEND_CANN_PACKAGE_PATH}/lib64" CACHE STRING "ASCEND library path")

    message(STATUS "Ascend device support enabled.")
    message(STATUS "RUN_MODE: ${RUN_MODE}")
    message(STATUS "SOC_VERSION: ${SOC_VERSION}")
    message(STATUS "ASCEND CANN package path: ${ASCEND_CANN_PACKAGE_PATH}")
    message(STATUS "ASCEND include path: ${ASCEND_CANN_INCLUDE_PATH}")
    message(STATUS "ASCEND library path: ${ASCEND_CANN_LIBRARY_PATH}")
    add_subdirectory(src/ascend/)

else()
    message(STATUS "Ascend device support disabled.")
endif()

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

# build shared library
add_library(minitorch SHARED ${MINITORCH_SRC_FILES})
target_include_directories(minitorch PUBLIC ${MINITORCH_INCLUDE_DIR})

if(USE_ASCEND)
    target_include_directories(minitorch PUBLIC

        # ${ASCEND_INCLUDE_PATH}
        ${ASCEND_CANN_INCLUDE_PATH}
    )

    target_link_directories(minitorch PRIVATE
        ${ASCEND_CANN_LIBRARY_PATH}
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
    -Wall # use all warnings
    -Wextra # use extra warnings
    -O2 # optimization level 2
    -std=c++17 # C++ 17 standard
    -g
)

# build test
add_executable(tensor_test ${CMAKE_CURRENT_SOURCE_DIR}/test/tensor_test.cpp)

# set target compile options
target_compile_options(tensor_test PUBLIC
    -Wall # use all warnings
    -Wextra # use extra warnings
    -O2 # optimization level 2
    -std=c++17 # C++ 17 standard
    -g
)

# link minitorch library
target_link_libraries(tensor_test minitorch)
