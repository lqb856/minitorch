/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-31 19:06:34
 * @Description  :
 */
#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <string>

namespace dlsys {
namespace runtime {

class Tensor;
class TensorContext;
class TensorFunction;

/**
 * @brief compute history of tensor,this is used to construct the backward graph
 */
class TensorHistory {
public:
  TensorHistory() {
    inputs_ = {};
    context_ = std::make_shared<TensorContext>(true);
    last_function_ = std::nullopt;
  }
  TensorHistory(std::vector<std::shared_ptr<Tensor>> inputs,
                std::shared_ptr<TensorContext> context,
                std::shared_ptr<TensorFunction> last_function)
      : inputs_(std::move(inputs)), context_(std::move(context)),
        last_function_(std::move(last_function)) {}
  ~TensorHistory() = default;
  std::string to_string() const;

  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::optional<std::shared_ptr<TensorContext>> context_;
  std::optional<std::shared_ptr<TensorFunction>> last_function_;
};

} // namespace runtime
} // namespace dlsys
