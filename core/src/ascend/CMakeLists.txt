cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

if(${RUN_MODE})
    set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
endif()
if (${SOC_VERSION})
    set(SOC_VERSION "Ascend310b4" CACHE STRING "system on chip type")
endif()
set(ASCEND_CANN_PACKAGE_PATH "/opt/ascend/ascend-toolkit/latest" CACHE STRING "ASCEND CANN package installation directory")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug"  CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

# if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
#     set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out"  CACHE STRING "path for install()" FORCE)
# endif()

file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_op_kernal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary_op_kernal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend_api_list.h
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_tiling.h
)

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()