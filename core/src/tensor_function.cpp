/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-01 15:58:29
 * @Description  :
 */

#include "tensor_function.h"
#include "tensor.h"
#include "tensor_backend_manager.h"
#include "tensor_helper.h"
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dlsys {
namespace runtime {

std::shared_ptr<Tensor>
TensorFunction::apply(const std::vector<std::shared_ptr<Tensor>> &inputs) {
  bool need_grad = false;
  for (auto &input : inputs) {
    if (input->requires_grad()) {
      need_grad = true;
      break;
    }
  }

  // create context
  auto ctx = std::make_shared<TensorContext>(need_grad);
  // forward computation
  auto output = this->forward(ctx, inputs);
  // save history
  if (need_grad) {
    output->history_ = std::make_shared<TensorHistory>(
        inputs, ctx, shared_from_this());
  }

  return output;
}

FORWARD_FUNCTION_IMPL(AddFunction) {
  assert(inputs.size() == 2);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_)
      .Add_Zip(inputs[0], inputs[1], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(AddFunction) { return {inputs, inputs}; }

FORWARD_FUNCTION_IMPL(NegFunction) {
  assert(inputs.size() == 1);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_).Neg_Map(inputs[0], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(NegFunction) {
  assert(inputs != nullptr);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs->ctx_).Neg_Map(inputs, output);
  return {output};
}

FORWARD_FUNCTION_IMPL(InvFunction) {
  assert(inputs.size() == 1);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_).Inv_Map(inputs[0], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(InvFunction) {
  assert(inputs != nullptr);
  std::shared_ptr<Tensor> output = nullptr;
  std::shared_ptr<Tensor> output1 = nullptr;
  // x^(-1)` = -1 / x^2
  TensorBackendManager::Get(inputs->ctx_).Sqrt_Map(inputs, output);
  TensorBackendManager::Get(inputs->ctx_).Inv_Map(output, output1);
  TensorBackendManager::Get(inputs->ctx_).Neg_Map(output1, output);
  return {output};
}

FORWARD_FUNCTION_IMPL(MulFunction) {
  assert(inputs.size() == 2);
  // TODO(lqb): save input is not necessary,
  // cause we will pass ctx to backward,
  // which contains inputs
  ctx->save_for_backward(inputs);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_)
      .Mul_Zip(inputs[0], inputs[1], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(MulFunction) {
  assert(inputs != nullptr);
  auto saved_tensors = ctx->saved_tensors();
  assert(saved_tensors[0] != nullptr);
  assert(saved_tensors[1] != nullptr);
  std::shared_ptr<Tensor> output1 = nullptr;
  std::shared_ptr<Tensor> output2 = nullptr;
  TensorBackendManager::Get(inputs->ctx_)
      .Mul_Zip(inputs, saved_tensors[1], output1);
  TensorBackendManager::Get(inputs->ctx_)
      .Mul_Zip(inputs, saved_tensors[0], output2);
  return {output1, output2};
}

FORWARD_FUNCTION_IMPL(SigmodFunction) {
  assert(inputs.size() == 1);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_).Sigmoid_Map(inputs[0], output);
  ctx->save_for_backward(output);
  return output;
}

BACKWARD_FUNCTION_IMPL(SigmodFunction) {
  auto saved_tensors = ctx->saved_tensors();
  std::shared_ptr<Tensor> output1 = nullptr;
  std::shared_ptr<Tensor> output2 = nullptr;
  TensorBackendManager::Get(inputs->ctx_).Neg_Map(saved_tensors[0], output1);
  TensorBackendManager::Get(inputs->ctx_)
      .Add_Zip(output1, tensor(inputs->ctx_, std::vector<float>{1.0}), output2);
  TensorBackendManager::Get(inputs->ctx_)
      .Mul_Zip(output2, saved_tensors[0], output1);
  TensorBackendManager::Get(inputs->ctx_).Mul_Zip(output1, inputs, output2);
  return {output2};
}

FORWARD_FUNCTION_IMPL(ReluFunction) {
  assert(inputs.size() == 1);
  ctx->save_for_backward(inputs);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_).Relu_Map(inputs[0], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(ReluFunction) {
  throw std::runtime_error("Not implemented");
}

FORWARD_FUNCTION_IMPL(LogFunction) {
  assert(inputs.size() == 1);
  ctx->save_for_backward(inputs);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_).Log_Map(inputs[0], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(LogFunction) {
  throw std::runtime_error("Not implemented");
}

FORWARD_FUNCTION_IMPL(ExpFunction) {
  assert(inputs.size() == 1);
  ctx->save_for_backward(inputs);
  std::shared_ptr<Tensor> output = nullptr;
  TensorBackendManager::Get(inputs[0]->ctx_).Exp_Map(inputs[0], output);
  return output;
}

BACKWARD_FUNCTION_IMPL(ExpFunction) { return {inputs}; }

FORWARD_FUNCTION_IMPL(SumFunction) {
  throw std::runtime_error("Not implemented");
}

BACKWARD_FUNCTION_IMPL(SumFunction) {
  throw std::runtime_error("Not implemented");
}

} // namespace runtime
} // namespace dlsys