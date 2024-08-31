# message("ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
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
    -O2 # 优化级别 2
    -std=c++17 # C++ 17 标准
    -g
)

target_include_directories(ascend_kernels_${RUN_MODE} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_INCLUDE_PATH}
)