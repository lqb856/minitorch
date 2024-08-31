/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-04 17:01:19
 * @Description  :
 */

#pragma once

#include "dl_context.h"
#include "tensor_backend.h"
#include "tensor_backend_atlas.h"
#include "tensor_backend_cpu.h"
#include "tensor_backend_gpu.h"
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dlsys {
namespace runtime {

class TensorBackendManager {
public:
  static const int kMaxDeviceAPI = 8;

  static TensorBackend &Get(DLContext ctx) {
    return Global()->GetAPI(ctx.device_type_);
  }

private:
  std::vector<TensorBackend *> api_{};

  TensorBackendManager() {
    api_.resize(kMaxDeviceAPI);
    std::fill(api_.begin(), api_.end(), nullptr);
    static TensorBackendCPU cpu_backend_inst;
    // static TensorBackendGPU gpu_backend_inst;
    static TensorBackendAtlas atlas_backend_inst;
    api_[static_cast<int>(DLDeviceType::kCPU)] = static_cast<TensorBackend *>(&cpu_backend_inst);
    api_[static_cast<int>(DLDeviceType::KAtlas)] = static_cast<TensorBackend *>(&atlas_backend_inst);
  }

  // Get global static variable.
  static TensorBackendManager *Global() {
    static TensorBackendManager inst;
    return &inst;
  }

  // Get API.
  TensorBackend &GetAPI(DLDeviceType type) {
    if (api_[static_cast<int>(type)] == nullptr) {
      std::cerr << "Device API not supported" << std::endl;
      exit(EXIT_FAILURE);
    }
    return *api_[static_cast<int>(type)];
  }
};

} // namespace runtime
} // namespace dlsys
