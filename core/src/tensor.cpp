/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-03 13:58:22
 * @Description  :
 */

#include "tensor.h"
#include "tensor_function.h"
#include <memory>
#include <vector>

namespace dlsys {
namespace runtime {

int Tensor::tensor_count_ = 0;

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs) {
  // Ensure lhs and rhs are not null
  if (!lhs || !rhs) {
    throw std::invalid_argument("Cannot add null Tensor pointers.");
  }

  std::shared_ptr<AddFunction> add_func = std::make_shared<AddFunction>();
  return add_func->apply({lhs, rhs});
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs) {
  // Ensure lhs and rhs are not null
  if (!lhs || !rhs) {
    throw std::invalid_argument("Cannot add null Tensor pointers.");
  }

  std::shared_ptr<NegFunction> neg_func = std::make_shared<NegFunction>();
  auto neged = neg_func->apply({rhs});
  std::shared_ptr<AddFunction> add_func = std::make_shared<AddFunction>();
  return add_func->apply({lhs, neged});
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs) {
  // Ensure lhs and rhs are not null
  if (!lhs || !rhs) {
    throw std::invalid_argument("Cannot mul null Tensor pointers.");
  }

  std::shared_ptr<MulFunction> mul_func = std::make_shared<MulFunction>();
  return mul_func->apply({lhs, rhs});
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs) {
  // Ensure lhs and rhs are not null
  if (!lhs || !rhs) {
    throw std::invalid_argument("Cannot div null Tensor pointers.");
  }

  std::shared_ptr<InvFunction> inv_func = std::make_shared<InvFunction>();
  auto inved = inv_func->apply({rhs});

  std::shared_ptr<MulFunction> mul_func = std::make_shared<MulFunction>();
  return mul_func->apply({lhs, inved});
}

} // namespace runtime
} // namespace dlsys