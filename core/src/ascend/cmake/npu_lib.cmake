message("ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascend_kernels_${RUN_MODE} SHARED
    ${KERNEL_FILES}
)

ascendc_compile_definitions(ascend_kernels_${RUN_MODE} PRIVATE 
  $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
  -DASCENDC_DUMP
  -DHAVE_WORKSPACE
  -DHAVE_TILING
)

target_compile_options(ascend_kernels_${RUN_MODE} PUBLIC 
    -Wall # 启用所有警告
    -Wextra # 启用额外警告
    -O1 # 优化级别 2
    -std=c++17 # C++ 17 标准
    -g
)

target_include_directories(ascend_kernels_${RUN_MODE} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

# target_link_libraries(ascend_kernels_${RUN_MODE} PUBLIC
#     $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
#     $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::${SOC_VERSION}>>
#     $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
#     $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:c_sec>>
#     tiling_api
#     register
#     platform
#     ascendalog
#     dl
# )

# # 指定库的安装目录
# install(TARGETS ascend_kernels_${RUN_MODE}
#     EXPORT ascend_kernels_${RUN_MODE}
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# # 仅安装特定的头文件到 include 目录
# install(FILES 
#     ${CMAKE_CURRENT_SOURCE_DIR}/ascend_api_list.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/custom_tiling.h
#     DESTINATION include/ascend_kernels_${RUN_MODE}
# )

# # 安装目标导出文件
# install(EXPORT ascend_kernels_${RUN_MODE}_target
#     FILE ascend_kernels_${RUN_MODE}.cmake
#     DESTINATION lib/cmake/ascend_kernels_${RUN_MODE}
# )