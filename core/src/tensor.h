/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-28 17:31:57
 * @Description  :
 */
#pragma once

#include "auto_grad.h"
#include "dl_context.h"
#include "tensor_context.h"
#include "tensor_data.h"
#include "tensor_function.h"
#include "tensor_history.h"
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// TODO(lqb): 实现 Tensor 类，用于表示张量， Tensor
namespace dlsys {
namespace runtime {

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  Tensor() = default;
  ~Tensor() = default;
  Tensor(DLContext ctx, const std::shared_ptr<TensorData> &data) {
    ctx_ = ctx;
    data_ = data;
    unique_id_ = Tensor::tensor_count_++;
    name_ = "tensor_" + std::to_string(unique_id_);
    grad_ = std::nullopt;
    history_ = std::nullopt;
  }

  /**
   * @brief is this tensor requires grad
   */
  bool requires_grad() const { return history_.has_value(); }

  /**
   * @brief set requires grad
   * @param requires_grad
   */
  void set_requires_grad(bool requires_grad) {
    if (requires_grad) {
      history_ = std::make_unique<TensorHistory>();
    } else {
      history_ = std::nullopt;
    }
  }

  /**
   * @brief get shape of the tensor
   * @return shape of the tensor
   */
  const std::vector<int> &shape() const { return data_->shape_; }

  std::vector<int> mutable_shape() { return data_->shape_; }

  const std::vector<int> &strides() const { return data_->strides_; }

  int size() const { return data_->size_; }

  int dims() const { return data_->n_dim_; }

  std::string name() const { return name_; }

  bool is_leaf() {
    return !history_.has_value() ||
           !history_.value()->last_function_.has_value();
  }

  bool is_constant() { return !history_.has_value(); }

  std::vector<std::shared_ptr<Tensor>> &parents() {
    assert(history_.has_value());
    return history_.value()->inputs_;
  }

  DLContext ctx() const { return ctx_; }

  // TODO(lqb): reload operator for tensor

  friend std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs);

  friend std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs);

  friend std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs);
  
  friend std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs);

  friend std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs);

  std::shared_ptr<Tensor> neg() {
    std::shared_ptr<NegFunction> neg = std::make_shared<NegFunction>();
    return neg->apply({shared_from_this()});
  }

  std::shared_ptr<Tensor> inv() {
    std::shared_ptr<InvFunction> inv = std::make_shared<InvFunction>();
    return inv->apply({shared_from_this()});
  }

  std::shared_ptr<Tensor> log() {
    std::shared_ptr<LogFunction> log = std::make_shared<LogFunction>();
    return log->apply({shared_from_this()});
  }

  std::shared_ptr<Tensor> exp() {
    std::shared_ptr<ExpFunction> exp = std::make_shared<ExpFunction>();
    return exp->apply({shared_from_this()});
  }

  std::shared_ptr<Tensor> relu() {
    std::shared_ptr<ReluFunction> relu = std::make_shared<ReluFunction>();
    return relu->apply({shared_from_this()});
  }

  std::shared_ptr<Tensor> sigmoid() {
    std::shared_ptr<SigmodFunction> sigmoid =
        std::make_shared<SigmodFunction>();
    return sigmoid->apply({shared_from_this()});
  }

  // TODO(lqb): backward propagation related functions
  std::vector<std::shared_ptr<Tensor>>
  chain_rule(std::shared_ptr<Tensor> input) {
    assert(history_.has_value());
    assert(history_.value()->last_function_.has_value());
    assert(history_.value()->context_.has_value());

    std::cout << "chain_rule: "
              << history_.value()->last_function_.value()->func_name()
              << std::endl;
    auto grad = history_.value()->last_function_.value()->backward(
        history_.value()->context_.value(), input);
    assert(grad.size() == parents().size());
    return grad;
  }

  void accumulate_grad(const std::shared_ptr<Tensor> &grad) {
    if (!grad_.has_value()) {
      grad_ = zeros({});
    }
    *grad_ = *grad_ + grad;
  }

  void zero_grad() {
    if (grad_.has_value()) {
      grad_.reset();
    }
  }

  std::shared_ptr<Tensor> grad() {
    assert(grad_.has_value());
    return grad_.value();
  }

  void backward(const std::shared_ptr<Tensor> &grad) {
    // assert(requires_grad());
    assert(grad->shape() == shape());
    assert(grad->ctx() == ctx());
    // assert(grad->data()->dtype_ == data_->dtype_);
    back_propagate(shared_from_this(), grad);
  }

  // TODO(lqb): tensor 的其他辅助函数

  // std::shared_ptr<Tensor> shared_from_this() {
  //   return std::shared_ptr<Tensor>(this);
  // }

  void to(const DLContext &ctx) {
    if (ctx_ == ctx)
      return;
    data_->to(ctx);
    ctx_ = ctx;
  }

  static std::shared_ptr<Tensor>
  make(DLContext ctx, const std::vector<int> &shape, void *data) {
    std::shared_ptr<Tensor> tensor;
    if (data == nullptr) {
      tensor = std::make_shared<Tensor>(
          ctx, std::make_shared<TensorData>(ctx, shape));
    } else {
      tensor = std::make_shared<Tensor>(
          ctx, std::make_shared<TensorData>(ctx, shape, data));
    }
    return tensor;
  }

  std::shared_ptr<Tensor> zeros(const std::vector<int> &shape) {
    if (shape.empty()) {
      return Tensor::make(this->ctx_, this->shape(), nullptr);
    }
    return Tensor::make(this->ctx_, shape, nullptr);
  }

  std::string to_string() {
    std::string ret = "Tensor(";
    ret += "name=" + name_ + ", ";
    ret += "ctx=" + ctx_.to_string() + ", ";
    ret += "history=" +
           (history_.has_value() ? history_.value()->to_string() : "None") +
           ", ";
    ret += "data=" + data_->to_string() + ", ";
    ret += "requires_grad=" + std::to_string(requires_grad()) + ", ";
    ret += "is_leaf=" + std::to_string(is_leaf()) + ", ";
    ret += "is_constant=" + std::to_string(is_constant()) + ", ";
    ret += "grad=" + (grad_.has_value() ? grad_.value()->to_string() : "None") +
           ")";
    return ret;
  }

  static int tensor_count_;
  int unique_id_;
  std::string name_;
  DLContext ctx_;
  std::shared_ptr<TensorData> data_;
  std::optional<std::shared_ptr<Tensor>> grad_;
  std::optional<std::shared_ptr<TensorHistory>> history_;
};

} // namespace runtime
} // namespace dlsys