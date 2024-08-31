/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-31 19:10:41
 * @Description  :
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace dlsys {
namespace runtime {

class Tensor;
class TensorContext;

enum class TensorFunctionType {
  Add,
  Neg,
  Inv,
  Mul,
  Sigmod,
  Relu,
  Log,
  Exp,
  Sum,
  // TODO(lqb): add more tensor function
};

/**
 * @brief base class of tensor function
 */
class TensorFunction : public std::enable_shared_from_this<TensorFunction> {
public:
  /**
   * @brief apply function, this method wraps forward method
   *  and construct the backward graph while forward
   * @param inputs activation from last layer
   * @return activation of this layer
   */
  std::shared_ptr<Tensor>
  apply(const std::vector<std::shared_ptr<Tensor>> &inputs);

  /**
   * @brief forward function, multiple inputs to generate one output
   * @note this is a pure virtual function, should be implemented by subclass
   * @param ctx context of this layer
   * @param inputs activation from last layer
   * @return activation of this layer
   */
  virtual std::shared_ptr<Tensor>
  forward(const std::shared_ptr<TensorContext> &ctx,
          const std::vector<std::shared_ptr<Tensor>> &inputs) = 0;

  /**
   * @brief backward function, one input to generate multiple outputs
   * @note this is a pure virtual function, should be implemented by subclass
   * @param ctx context of this layer
   * @param grad_back gradient from next layer
   * @return gradient of this layer
   */
  virtual std::vector<std::shared_ptr<Tensor>>
  backward(const std::shared_ptr<TensorContext> &ctx,
           const std::shared_ptr<Tensor> &inputs) = 0;

  virtual std::string func_name() = 0;

  virtual ~TensorFunction() = default;
};

/**
 * @brief register tensor function
 * @param name name of the tensor function
 */
#define REGISTER_TENSOR_FUNCTION(name)                                         \
  class name : public TensorFunction {                                         \
  public:                                                                      \
    name() = default;                                                          \
    std::shared_ptr<Tensor>                                                    \
    forward(const std::shared_ptr<TensorContext> &ctx,                         \
            const std::vector<std::shared_ptr<Tensor>> &inputs) override;      \
    std::vector<std::shared_ptr<Tensor>>                                       \
    backward(const std::shared_ptr<TensorContext> &ctx,                        \
             const std::shared_ptr<Tensor> &grad_back) override;               \
    std::string func_name() override { return #name; }                         \
  };

#define FORWARD_FUNCTION_IMPL(name)                                            \
  std::shared_ptr<Tensor> name::forward(                                       \
      const std::shared_ptr<TensorContext> &ctx,                               \
      const std::vector<std::shared_ptr<Tensor>> &inputs)

#define BACKWARD_FUNCTION_IMPL(name)                                           \
  std::vector<std::shared_ptr<Tensor>> name::backward(                         \
      const std::shared_ptr<TensorContext> &ctx,                               \
      const std::shared_ptr<Tensor> &inputs)

REGISTER_TENSOR_FUNCTION(AddFunction);
REGISTER_TENSOR_FUNCTION(SubFunction);
REGISTER_TENSOR_FUNCTION(MulFunction);
// REGISTER_TENSOR_FUNCTION(DivFunction);
REGISTER_TENSOR_FUNCTION(NegFunction);
REGISTER_TENSOR_FUNCTION(InvFunction);
REGISTER_TENSOR_FUNCTION(SigmodFunction);
REGISTER_TENSOR_FUNCTION(ReluFunction);
REGISTER_TENSOR_FUNCTION(LogFunction);
REGISTER_TENSOR_FUNCTION(ExpFunction);
REGISTER_TENSOR_FUNCTION(SumFunction);
// TODO(lqb): add more tensor function

} // namespace runtime
} // namespace dlsys