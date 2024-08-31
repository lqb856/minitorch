/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-28 15:52:43
 * @Description  :
 */

#pragma once

// FIXME(lqb): 将实现放在 cpp 文件里时，会找不到
// "acl/acl.h"。因此，将实现放在头文件里，并使用了 inline 关键字
#include "acl/acl.h"
#include "custom_tiling.h"
#include "aclrtlaunch_binary_op_kernal_float.h"
#include "aclrtlaunch_binary_op_kernal_half.h"
#include "aclrtlaunch_binary_op_kernal_int32_t.h"
#include "aclrtlaunch_unary_op_kernal_float.h"
#include "aclrtlaunch_unary_op_kernal_half.h"
#include "aclrtlaunch_unary_op_kernal_int32_t.h"
#include <cstddef>
#include <cstdint>
#include <iostream>

// ascend 310b4 has 8 cores
const int kMaxBlockDim = 8;

#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
    }                                                                          \
    exit(-1);                                                                  \
  } while (0);

// TODO(lqb): add more kernel functions
// TODO(lqb): wrap the kernel functions to avoid the user to call the kernel
// functions directly call of kernel function
inline void binary_op_half(uint32_t blockDim, void *l2ctrl, void *stream,
                           uint8_t *x, uint8_t *y, uint8_t *z,
                           CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(binary_op_kernal_half)
  (blockDim, stream, x, y, z, nullptr, tiling);
}

inline void binary_op_float(uint32_t blockDim, void *l2ctrl, void *stream,
                            uint8_t *x, uint8_t *y, uint8_t *z,
                            CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(binary_op_kernal_float)
  (blockDim, stream, x, y, z, nullptr, tiling);
}

inline void binary_op_int32_t(uint32_t blockDim, void *l2ctrl, void *stream,
                              uint8_t *x, uint8_t *y, uint8_t *z,
                              CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(binary_op_kernal_int32_t)
  (blockDim, stream, x, y, z, nullptr, tiling);
}

inline void unary_op_half(uint32_t blockDim, void *l2ctrl, void *stream,
                          uint8_t *x, uint8_t *y, CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(unary_op_kernal_half)
  (blockDim, stream, x, y, nullptr, nullptr, tiling);
}

inline void unary_op_float(uint32_t blockDim, void *l2ctrl, void *stream,
                           uint8_t *x, uint8_t *y, CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(unary_op_kernal_float)
  (blockDim, stream, x, y, nullptr, nullptr, tiling);
}

inline void unary_op_int32_t(uint32_t blockDim, void *l2ctrl, void *stream,
                             uint8_t *x, uint8_t *y, CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(unary_op_kernal_int32_t)
  (blockDim, stream, x, y, nullptr, nullptr, tiling);
}

inline void add_float(float *x, float *y, float *z, size_t size, void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::ADD);
  tiling.totalLength = size * sizeof(float);
  tiling.tileNum = 1;
  binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                  (uint8_t *)z, &tiling);
}

inline void add_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
                        void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::ADD);
  tiling.totalLength = size * sizeof(int32_t);
  tiling.tileNum = 1;
  binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                    (uint8_t *)z, &tiling);
}

inline void sub_float(float *x, float *y, float *z, size_t size, void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::SUB);
  tiling.totalLength = size * sizeof(float);
  tiling.tileNum = 1;
  binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                  (uint8_t *)z, &tiling);
}

inline void sub_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
                        void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::SUB);
  tiling.totalLength = size * sizeof(int32_t);
  tiling.tileNum = 1;
  binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                    (uint8_t *)z, &tiling);
}

inline void mul_float(float *x, float *y, float *z, size_t size, void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::MUL);
  tiling.totalLength = size * sizeof(float);
  tiling.tileNum = 1;
  binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                  (uint8_t *)z, &tiling);
}

inline void mul_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
                        void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::MUL);
  tiling.totalLength = size * sizeof(int32_t);
  tiling.tileNum = 1;
  binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                    (uint8_t *)z, &tiling);
}

inline void div_float(float *x, float *y, float *z, size_t size, void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::DIV);
  tiling.totalLength = size * sizeof(float);
  tiling.tileNum = 1;
  binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                  (uint8_t *)z, &tiling);
}

inline void div_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
                        void *stream) {
  // TODO(lqb): add support for large size
  CustomTilingData tiling;
  tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::DIV);
  tiling.totalLength = size * sizeof(int32_t);
  tiling.tileNum = 1;
  binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
                    (uint8_t *)z, &tiling);
}

// TODO(lqb): add more binary operations

// TODO(lqb): add more unary operations

inline void *alloc_data_space(size_t size) {
  void *ptr;
  CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  return ptr;
}

inline void free_data_space(void *ptr) { CHECK_ACL(aclrtFree(ptr)); }

inline void copy_data_host_to_dev(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

inline void copy_data_dev_to_host(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

inline void copy_data_host_to_host(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_HOST_TO_HOST));
}

inline void copy_data_dev_to_dev(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
}

inline void *stream_create() {
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  return static_cast<void *>(stream);
}

inline void stream_destroy(void *stream) {
  CHECK_ACL(aclrtDestroyStream(static_cast<aclrtStream>(stream)));
}

inline void stream_sync(void *stream) {
  CHECK_ACL(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
}

inline void *context_create(int device_id) {
  aclrtContext context;
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(device_id));
  CHECK_ACL(aclrtCreateContext(&context, device_id));
  return static_cast<void *>(context);
}

inline void context_destroy(void *context, int device_id) {
  CHECK_ACL(aclrtDestroyContext(static_cast<aclrtContext>(context)));
  CHECK_ACL(aclrtResetDevice(device_id));
  CHECK_ACL(aclFinalize());
}