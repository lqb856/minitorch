// /*
//  * @Author       : Li Qingbing(3263109808@qq.com)
//  * @Version      : V0.0
//  * @Date         : 2024-07-29 15:33:18
//  * @Description  :
//  */

// #ifndef DLSYS_RUNTIME_CUDA_DEVICE_API_H_
// #define DLSYS_RUNTIME_CUDA_DEVICE_API_H_

// #include "device_api.h"
// #include <cuda_runtime.h>

// #include <assert.h>
// #include <string>

// namespace dlsys {
// namespace runtime {

// class CUDADeviceAPI : public DeviceAPI {
// public:
//   void *AllocDataSpace(DLContext ctx, size_t size, size_t alignment) final;
//   void FreeDataSpace(DLContext ctx, void *ptr) final;
//   void CopyDataFromTo(const void *from, void *to, size_t size,
//                       DLContext ctx_from, DLContext ctx_to,
//                       DLStreamHandle stream) final;
//   void StreamSync(DLContext ctx, DLStreamHandle stream) final;
// };

// } // namespace runtime
// } // namespace dlsys

// #endif