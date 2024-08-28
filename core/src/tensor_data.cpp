/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-05 13:09:32
 * @Description  :
 */

#include "tensor_data.h"
#include "device_api_manager.h"

namespace dlsys {
namespace runtime {

TensorData::TensorData(const DLContext ctx, const std::vector<int> shape) {
  ctx_ = ctx;
  shape_ = shape;
  n_dim_ = shape.size();
  size_ = 1;
  strides_.resize(n_dim_);
  for (int i = n_dim_ - 1; i >= 0; --i) {
    strides_[i] = size_;
    size_ *= shape[i];
  }
  data_ = DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, size_ * sizeof(float),
                                                     64);
}

TensorData::TensorData(const DLContext ctx, const std::vector<int> shape,
                       void *data) {
  ctx_ = ctx;
  shape_ = shape;
  n_dim_ = shape.size();
  size_ = 1;
  strides_.resize(n_dim_);
  for (int i = n_dim_ - 1; i >= 0; --i) {
    strides_[i] = size_;
    size_ *= shape[i];
  }
  data_ = DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, size_ * sizeof(float),
                                                     64);
  DeviceAPIManager::Get(ctx)->CopyDataFromTo(data, data_, size_ * sizeof(float),
                                             ctx, ctx, nullptr);
}

TensorData::~TensorData() {
  if (data_ != nullptr)
    DeviceAPIManager::Get(ctx_)->FreeDataSpace(ctx_, data_);
}

void TensorData::to(const DLContext &new_ctx) {
  if (ctx_ == new_ctx)
    return;
  void *new_data = DeviceAPIManager::Get(new_ctx)->AllocDataSpace(
      new_ctx, size_ * sizeof(float), 64);
  DeviceAPIManager::Get(new_ctx)->CopyDataFromTo(
      data_, new_data, size_ * sizeof(float), ctx_, new_ctx, nullptr);
  DeviceAPIManager::Get(ctx_)->FreeDataSpace(ctx_, data_);
  data_ = new_data;
  ctx_ = new_ctx;
}

std::string TensorData::to_string() const {
  std::string ret = "TensorData(";
  ret += "shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    ret += std::to_string(shape_[i]);
    if (i != shape_.size() - 1)
      ret += ", ";
  }
  ret += "], ";
  ret += "strides=[";
  for (size_t i = 0; i < strides_.size(); ++i) {
    ret += std::to_string(strides_[i]);
    if (i != strides_.size() - 1)
      ret += ", ";
  }
  ret += "], ";
  ret += "data=[";
  for (int i = 0; i < size_; ++i) {
    ret += std::to_string(static_cast<float *>(data_)[i]);
    if (i != size_ - 1)
      ret += ", ";
  }
  ret += "])";
  return ret;
}

} // namespace runtime
} // namespace dlsys