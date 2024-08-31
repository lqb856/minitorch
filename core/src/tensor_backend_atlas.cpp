/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-30 13:43:59
 * @Description  :
 */

#include "tensor_backend_atlas.h"
#include "ascend_api_list.h"
#include "stream_manager.h"
#include "tensor.h"
#include <cassert>
#include <stdexcept>

namespace dlsys {
namespace runtime {

void TensorBackendAtlas::Neg_Map(const std::shared_ptr<Tensor> &a,
                                 std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Inv_Map(const std::shared_ptr<Tensor> &a,
                                 std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Log_Map(const std::shared_ptr<Tensor> &a,
                                 std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Sqrt_Map(const std::shared_ptr<Tensor> &a,
                                  std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Pow_Map(const std::shared_ptr<Tensor> &a,
                                 std::shared_ptr<Tensor> &c) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Exp_Map(const std::shared_ptr<Tensor> &a,
                                 std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Relu_Map(const std::shared_ptr<Tensor> &a,
                                  std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Sigmoid_Map(const std::shared_ptr<Tensor> &a,
                                     std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Add_Zip(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b,
                                 std::shared_ptr<Tensor> &c) {
  assert(a != nullptr && b != nullptr);
  // TODO(lqb): shape broadcast
  // now we just assume a and b have the same shape
  assert(a->shape() == b->shape());
  if (c == nullptr) {
    c = a->zeros({});
  } else {
    assert(c->shape() == a->shape());
  }

  add_float(static_cast<float *>(a->data_->data_),
            static_cast<float *>(b->data_->data_),
            static_cast<float *>(c->data_->data_), a->size(), StreamManager::GetStream());
}

void TensorBackendAtlas::Sub_Zip(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b,
                                 std::shared_ptr<Tensor> &c) {
  assert(a != nullptr && b != nullptr);
  // TODO(lqb): shape broadcast
  // now we just assume a and b have the same shape
  assert(a->shape() == b->shape());
  if (c == nullptr) {
    c = a->zeros({});
  } else {
    assert(c->shape() == a->shape());
  }

  sub_float(static_cast<float *>(a->data_->data_),
            static_cast<float *>(b->data_->data_),
            static_cast<float *>(c->data_->data_), a->size(), StreamManager::GetStream());
}

void TensorBackendAtlas::Mul_Zip(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b,
                                 std::shared_ptr<Tensor> &c) {
  assert(a != nullptr && b != nullptr);
  // TODO(lqb): shape broadcast
  // now we just assume a and b have the same shape
  assert(a->shape() == b->shape());
  if (c == nullptr) {
    c = a->zeros({});
  } else {
    assert(c->shape() == a->shape());
  }

  mul_float(static_cast<float *>(a->data_->data_),
            static_cast<float *>(b->data_->data_),
            static_cast<float *>(c->data_->data_), a->size(), StreamManager::GetStream());
}

void TensorBackendAtlas::Div_Zip(const std::shared_ptr<Tensor> &a,
                                 const std::shared_ptr<Tensor> &b,
                                 std::shared_ptr<Tensor> &c) {
  assert(a != nullptr && b != nullptr);
  // TODO(lqb): shape broadcast
  // now we just assume a and b have the same shape
  assert(a->shape() == b->shape());
  if (c == nullptr) {
    c = a->zeros({});
  } else {
    assert(c->shape() == a->shape());
  }

  div_float(static_cast<float *>(a->data_->data_),
            static_cast<float *>(b->data_->data_),
            static_cast<float *>(c->data_->data_), a->size(), StreamManager::GetStream());
}

void TensorBackendAtlas::Sum_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                                    std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Mean_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                                     std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::Mul_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                                    std::shared_ptr<Tensor> &b) {
  throw std::runtime_error("Not implemented");
}

void TensorBackendAtlas::MatMul(const std::shared_ptr<Tensor> &a,
                                const std::shared_ptr<Tensor> &b,
                                std::shared_ptr<Tensor> &c) {
  throw std::runtime_error("Not implemented");
}

} // namespace runtime
} // namespace dlsys