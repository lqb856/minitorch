/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-29 16:15:05
 * @Description  : 
 */

#pragma once

#include "cpu_device_api.h"
#include "ascend_device_api.h"
#include "dl_context.h"
// #include "cuda_device_api.h"
#include <iostream>
#include <array>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dlsys {
namespace runtime {

class DeviceAPIManager {
public:
  static const int kMaxDeviceAPI = 8;

  static DeviceAPI *Get(DLContext ctx) {
    return Global()->GetAPI(ctx.device_type_);
  }

private:
  std::vector<DeviceAPI *> api_{};

  DeviceAPIManager() {
    api_.resize(kMaxDeviceAPI);
    std::fill(api_.begin(), api_.end(), nullptr);
    static CPUDeviceAPI cpu_device_api_inst;
    static AscendDeviceAPI ascend_device_api_inst;
    // static CUDADeviceAPI gpu_device_api_inst;
    api_[static_cast<int>(DLDeviceType::kCPU)] = static_cast<DeviceAPI *>(&cpu_device_api_inst);
    api_[static_cast<int>(DLDeviceType::KAtlas)] = static_cast<DeviceAPI *>(&ascend_device_api_inst);
  }

  // Get global static variable.
  static DeviceAPIManager *Global() {
    static DeviceAPIManager inst;
    return &inst;
  }

  // Get API.
  DeviceAPI *GetAPI(DLDeviceType type) {
    if (api_[static_cast<int>(type)] == nullptr) {
      std::cerr << "Device API not supported" << std::endl;
      exit(EXIT_FAILURE);
    }
    return api_[static_cast<int>(type)];
  }
};

} // namespace runtime
} // namespace dlsys