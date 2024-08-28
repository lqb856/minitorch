/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-01 15:12:42
 * @Description  : 
 */

#pragma once

#include <memory>
#include <vector>

namespace dlsys {
namespace runtime {

class Tensor;

/**
 * @brief save tensors for backward
 */
class TensorContext {
public:
  TensorContext(bool need_grad = false): need_grad_(need_grad) {};
  ~TensorContext() = default;

  /**
   * @brief save tensor for backward
   * @param tensor
   */
  void save_for_backward(const std::shared_ptr<Tensor> &tensor) {
    if (tensor == nullptr) return;
    if (!need_grad_) return;
    saved_values_.push_back(tensor);
  }

  /**
   * @brief save tensors for backward
   * @param tensors
   */
  void save_for_backward(const std::vector<std::shared_ptr<Tensor>> &tensors) {
    if (!need_grad_) return;
    saved_values_.insert(saved_values_.end(), tensors.begin(), tensors.end());
  }

  /**
   * @brief get saved tensors
   * @return saved tensors
   */
  std::vector<std::shared_ptr<Tensor>> saved_tensors() {
    return saved_values_;
  }

  bool need_grad_ = false;
  std::vector<std::shared_ptr<Tensor>> saved_values_;
};

} // namespace runtime
} // namespace dlsys