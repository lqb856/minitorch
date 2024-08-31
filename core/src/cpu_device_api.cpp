/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-29 15:36:03
 * @Description  :
 */

#include "cpu_device_api.h"

#include <cstdlib>
#include <cstring>

namespace dlsys {
namespace runtime {

void *CPUDeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                   size_t alignment) {
  void *ptr;
  int ret = posix_memalign(&ptr, alignment, size);
  if (ret != 0)
    throw std::bad_alloc();
  return ptr;
}
void CPUDeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) { free(ptr); }

void CPUDeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                  DLContext ctx_from, DLContext ctx_to,
                                  DLStreamHandle stream) {
  memcpy(to, from, size);
}

void CPUDeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {}

void CPUDeviceAPI::StreamCreate(DLContext ctx, DLStreamHandle *stream) {}

void CPUDeviceAPI::StreamDestroy(DLContext ctx, DLStreamHandle stream) {}

void CPUDeviceAPI::ContextCreate(DLContext ctx, DLContextHandle *context) {}

void CPUDeviceAPI::ContextDestroy(DLContext ctx, DLContextHandle context) {}

} // namespace runtime
} // namespace dlsys