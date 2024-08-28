cmake_minimum_required(VERSION 3.0)
project(AscendC_Kernels)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# set default values
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/opt/ascend/ascend-toolkit/8.0/" CACHE STRING "ASCEND CANN package installation directory")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend_api_list.h
)

message("ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels_${RUN_MODE} SHARED
    ${KERNEL_FILES}
)

ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE 
  $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
  -DASCENDC_DUMP
  -DHAVE_WORKSPACE
  -DHAVE_TILING
  )

target_include_directories(ascendc_kernels_${RUN_MODE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ascendc_kernels_${RUN_MODE} PRIVATE
    # $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    # $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::${SOC_VERSION}>>
    # $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    # $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:c_sec>>
    tiling_api
    register
    platform
    ascendalog
    dl
)

