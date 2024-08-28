/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-28 16:05:34
 * @Description  : 
 */

#pragma once

#include "ascend/ascend_api_list.h"
#include "ascend_device_api.h"
#include "dl_context.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace dlsys {
namespace runtime {

void *AscendDeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                   size_t alignment) {
  assert(ctx.device_type_ == DLDeviceType::KAtlas);
  return alloc_data_space(size);
}
void AscendDeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) { 
  assert(ctx.device_type_ == DLDeviceType::KAtlas);
  free_data_space(ptr);
}

void AscendDeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                  DLContext ctx_from, DLContext ctx_to,
                                  DLStreamHandle stream) {
  if (ctx_from.device_type_ == DLDeviceType::KAtlas &&
      ctx_to.device_type_ == DLDeviceType::KAtlas) {
    copy_data_dev_to_dev(from, to, size);
  } else if (ctx_from.device_type_ == DLDeviceType::KAtlas &&
             ctx_to.device_type_ == DLDeviceType::kCPU) {
    copy_data_dev_to_host(from, to, size);
  } else if (ctx_from.device_type_ == DLDeviceType::kCPU &&
             ctx_to.device_type_ == DLDeviceType::KAtlas) {
    copy_data_host_to_dev(from, to, size);
  } else if (ctx_from.device_type_ == DLDeviceType::kCPU &&
             ctx_to.device_type_ == DLDeviceType::kCPU) {
    copy_data_host_to_host(from, to, size);
  } else {
    assert(false);
  }
}

void AscendDeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {
  assert(ctx.device_type_ == DLDeviceType::KAtlas);
  stream_sync(stream);
}

} // namespace runtime
} // namespace dlsys
