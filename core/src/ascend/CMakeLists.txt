cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_op_kernal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary_op_kernal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend_api_list.h
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_tiling.h
)

# file(GLOB_RECURSE KERNEL_FILES
#     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/*.h
# )

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()