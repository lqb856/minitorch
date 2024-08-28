/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-03 14:09:00
 * @Description  :
 */

#pragma once

#include <memory>
#include <vector>
#include "dl_context.h"

namespace dlsys {
namespace runtime {

class TensorData {
public:
  TensorData() = default;
  TensorData(const DLContext ctx, const std::vector<int> shape);
  TensorData(const DLContext ctx, const std::vector<int> shape, void *data);
  ~TensorData();

  int n_dim() const { return n_dim_; }
  int size() const { return size_; }
  void *data() const { return data_; }
  DLContext ctx() const { return ctx_; }
  const std::vector<int> &shape() const { return shape_; }
  const std::vector<int> &strides() const { return strides_; }
  const float *data_ptr() const { return static_cast<float *>(data_); }
  float *mutable_data_ptr() const { return static_cast<float *>(data_); }
  void to(const DLContext &ctx);
  
  std::string to_string() const;
  
  int n_dim_;
  int size_;
  void *data_; // consider encapsulating this into TensorStorage class
  DLContext ctx_;
  std::vector<int> shape_;
  std::vector<int> strides_;
};

} // namespace runtime
} // namespace dlsys