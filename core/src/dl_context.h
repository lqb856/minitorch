/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-05 13:12:27
 * @Description  :
 */

#pragma once

#include <string>

enum class DLDeviceType {
  kCPU = 1,   // CPU
  kGPU = 2,   // GPU with CUDA
  KAtlas = 3, // Huawei Ascend 310
};

class DLContext {
public:
  int device_id_;
  DLDeviceType device_type_;

  std::string to_string() const {
    return "DLContext(device_id=" + std::to_string(device_id_) +
           ", device_type=" + std::to_string(static_cast<int>(device_type_)) +
           ")";
  }

  bool operator==(const DLContext &ctx) {
    return device_id_ == ctx.device_id_ && device_type_ == ctx.device_type_;
  }
};

typedef void *DLStreamHandle;