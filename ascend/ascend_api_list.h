/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-28 15:52:43
 * @Description  :
 */

#pragma once

#include "acl/acl.h"
#include "aclrtlaunch_binary_op.h"
#include "aclrtlaunch_unary_op.h"
#include "custom_tiling.h"
#include <cstdint>

#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
    }                                                                          \
    exit(-1);                                                                  \
  } while (0);

// call of kernel function
void binary_op_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
                  uint8_t *y, uint8_t *z, CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(binary_op)(blockDim, stream, x, y, z, nullptr, tiling);
}

void unary_op_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
                 uint8_t *y, CustomTilingData *tiling) {
  ACLRT_LAUNCH_KERNEL(unary_op)
  (blockDim, stream, x, y, nullptr, nullptr, tiling);
}

void *alloc_data_space(size_t size) {
  void *ptr;
  CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  return ptr;
}

void free_data_space(void *ptr) { CHECK_ACL(aclrtFree(ptr)); }

void copy_data_host_to_dev(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

void copy_data_dev_to_host(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

void copy_data_host_to_host(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_HOST_TO_HOST));
}

void copy_data_dev_to_dev(const void *from, void *to, size_t size) {
  CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
}

void *stream_create() {
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  return static_cast<void *>(stream);
}

void stream_destroy(void *stream) { CHECK_ACL(aclrtDestroyStream(static_cast<aclrtStream>(stream))); }

void stream_sync(void *stream) { CHECK_ACL(aclrtSynchronizeStream(static_cast<aclrtStream>(stream))); }

void *context_create(int device_id) {
  aclrtContext context;
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(device_id));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
  return static_cast<void *>(context);
}

void context_destroy(void *context, int device_id) {
  CHECK_ACL(aclrtDestroyContext(static_cast<aclrtContext>(context)));
  CHECK_ACL(aclrtResetDevice(device_id));
  CHECK_ACL(aclFinalize());
}