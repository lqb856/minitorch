/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-30 21:41:38
 * @Description  : 
 */

// #include "ascend_api_list.h"

// // TODO(lqb): add more kernel functions
// // TODO(lqb): wrap the kernel functions to avoid the user to call the kernel
// // functions directly call of kernel function
// void binary_op_half(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                     uint8_t *y, uint8_t *z, CustomTilingData *tiling) {
//   ACLRT_LAUNCH_KERNEL(binary_op_kernal_half)
//   (blockDim, stream, x, y, z, nullptr, tiling);
// }

// void binary_op_float(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                      uint8_t *y, uint8_t *z, CustomTilingData *tiling) {
//   ACLRT_LAUNCH_KERNEL(binary_op_kernal_float)
//   (blockDim, stream, x, y, z, nullptr, tiling);
// }

// void binary_op_int32_t(uint32_t blockDim, void *l2ctrl, void *stream,
//                        uint8_t *x, uint8_t *y, uint8_t *z,
//                        CustomTilingData *tiling) {
//   ACLRT_LAUNCH_KERNEL(binary_op_kernal_int32_t)
//   (blockDim, stream, x, y, z, nullptr, tiling);
// }

// void unary_op_half(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                    uint8_t *y, CustomTilingData *tiling) {
//   ACLRT_LAUNCH_KERNEL(unary_op_kernal_half)
//   (blockDim, stream, x, y, nullptr, nullptr, tiling);
// }

// void unary_op_float(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                     uint8_t *y, CustomTilingData *tiling) {
//   ACLRT_LAUNCH_KERNEL(unary_op_kernal_float)
//   (blockDim, stream, x, y, nullptr, nullptr, tiling);
// }

// void unary_op_int32_t(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                       uint8_t *y, CustomTilingData *tiling) {
//   ACLRT_LAUNCH_KERNEL(unary_op_kernal_int32_t)
//   (blockDim, stream, x, y, nullptr, nullptr, tiling);
// }

// void add_float(float *x, float *y, float *z, size_t size, void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::ADD);
//   tiling.totalLength = size * sizeof(float);
//   tiling.tileNum = 1;
//   binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                   (uint8_t *)z, &tiling);
// }

// void add_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
//                  void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::ADD);
//   tiling.totalLength = size * sizeof(int32_t);
//   tiling.tileNum = 1;
//   binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                     (uint8_t *)z, &tiling);
// }

// void sub_float(float *x, float *y, float *z, size_t size, void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::SUB);
//   tiling.totalLength = size * sizeof(float);
//   tiling.tileNum = 1;
//   binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                   (uint8_t *)z, &tiling);
// }

// void sub_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
//                  void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::SUB);
//   tiling.totalLength = size * sizeof(int32_t);
//   tiling.tileNum = 1;
//   binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                     (uint8_t *)z, &tiling);
// }

// void mul_float(float *x, float *y, float *z, size_t size, void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::MUL);
//   tiling.totalLength = size * sizeof(float);
//   tiling.tileNum = 1;
//   binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                   (uint8_t *)z, &tiling);
// }

// void mul_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
//                  void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::MUL);
//   tiling.totalLength = size * sizeof(int32_t);
//   tiling.tileNum = 1;
//   binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                     (uint8_t *)z, &tiling);
// }

// void div_float(float *x, float *y, float *z, size_t size, void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::DIV);
//   tiling.totalLength = size * sizeof(float);
//   tiling.tileNum = 1;
//   binary_op_float(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                   (uint8_t *)z, &tiling);
// }

// void div_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size,
//                  void *stream) {
//   // TODO(lqb): add support for large size
//   CustomTilingData tiling;
//   tiling.opType = static_cast<uint32_t>(NPU_OP_TYPE::DIV);
//   tiling.totalLength = size * sizeof(int32_t);
//   tiling.tileNum = 1;
//   binary_op_int32_t(kMaxBlockDim, nullptr, stream, (uint8_t *)x, (uint8_t *)y,
//                     (uint8_t *)z, &tiling);
// }

// // TODO(lqb): add more binary operations

// // TODO(lqb): add more unary operations

// void *alloc_data_space(size_t size) {
//   void *ptr;
//   CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
//   return ptr;
// }

// void free_data_space(void *ptr) { CHECK_ACL(aclrtFree(ptr)); }

// void copy_data_host_to_dev(const void *from, void *to, size_t size) {
//   CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_HOST_TO_DEVICE));
// }

// void copy_data_dev_to_host(const void *from, void *to, size_t size) {
//   CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_DEVICE_TO_HOST));
// }

// void copy_data_host_to_host(const void *from, void *to, size_t size) {
//   CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_HOST_TO_HOST));
// }

// void copy_data_dev_to_dev(const void *from, void *to, size_t size) {
//   CHECK_ACL(aclrtMemcpy(to, size, from, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
// }

// void *stream_create() {
//   aclrtStream stream;
//   CHECK_ACL(aclrtCreateStream(&stream));
//   return static_cast<void *>(stream);
// }

// void stream_destroy(void *stream) {
//   CHECK_ACL(aclrtDestroyStream(static_cast<aclrtStream>(stream)));
// }

// void stream_sync(void *stream) {
//   CHECK_ACL(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
// }

// void *context_create(int device_id) {
//   aclrtContext context;
//   CHECK_ACL(aclInit(nullptr));
//   CHECK_ACL(aclrtSetDevice(device_id));
//   CHECK_ACL(aclrtCreateContext(&context, device_id));
//   return static_cast<void *>(context);
// }

// void context_destroy(void *context, int device_id) {
//   CHECK_ACL(aclrtDestroyContext(static_cast<aclrtContext>(context)));
//   CHECK_ACL(aclrtResetDevice(device_id));
//   CHECK_ACL(aclFinalize());
// }




// // TODO(lqb): add more kernel functions
// // TODO(lqb): wrap the kernel functions to avoid the user to call the kernel
// // functions directly call of kernel function
// void binary_op_half(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                     uint8_t *y, uint8_t *z, CustomTilingData *tiling);

// void binary_op_float(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                      uint8_t *y, uint8_t *z, CustomTilingData *tiling);

// void binary_op_int32_t(uint32_t blockDim, void *l2ctrl, void *stream,
//                        uint8_t *x, uint8_t *y, uint8_t *z,
//                        CustomTilingData *tiling);

// void unary_op_half(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                    uint8_t *y, CustomTilingData *tiling);

// void unary_op_float(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                     uint8_t *y, CustomTilingData *tiling);

// void unary_op_int32_t(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x,
//                       uint8_t *y, CustomTilingData *tiling);

// void add_float(float *x, float *y, float *z, size_t size, void *stream);

// void add_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size, void *stream);

// void sub_float(float *x, float *y, float *z, size_t size, void *stream);

// void sub_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size, void *stream);

// void mul_float(float *x, float *y, float *z, size_t size, void *stream);

// void mul_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size, void *stream);

// void div_float(float *x, float *y, float *z, size_t size, void *stream);

// void div_int32_t(int32_t *x, int32_t *y, int32_t *z, size_t size, void *stream);

// // TODO(lqb): add more binary operations

// // TODO(lqb): add more unary operations

// void *alloc_data_space(size_t size);
// void free_data_space(void *ptr);

// void copy_data_host_to_dev(const void *from, void *to, size_t size);
// void copy_data_dev_to_host(const void *from, void *to, size_t size);
// void copy_data_host_to_host(const void *from, void *to, size_t size);
// void copy_data_dev_to_dev(const void *from, void *to, size_t size);

// void *stream_create();
// void stream_destroy(void *stream);
// void stream_sync(void *stream);
// void *context_create(int device_id);
// void context_destroy(void *context, int device_id);