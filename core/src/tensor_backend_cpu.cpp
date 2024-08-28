/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-06 16:07:16
 * @Description  :
 */

#include "tensor_backend_cpu.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <functional>
#include <vector>

namespace dlsys {
namespace runtime {

void TensorBackendCPU::map(const std::shared_ptr<Tensor> &a,
                           std::shared_ptr<Tensor> &b,
                           std::function<float(float)> f) {
  assert(a != nullptr);
  // is 'b' is nullptr, create a new tensor with the same shape as 'a'
  // if 'b' is not nullptr, check if the shape can be broadcasted
  if (b == nullptr) {
    b = a->zeros({});
  } else {
    std::vector<int> broadcasted_shape;
    assert(broadcast_shape(a->shape(), b->shape(), broadcasted_shape));
    assert(b->shape() == broadcasted_shape);
  }

  std::vector<int> in_index(a->shape().size(), 0);
  std::vector<int> out_index(b->shape().size(), 0);
  const float *a_data = a->data_->data_ptr();
  float *b_data = b->data_->mutable_data_ptr();

  for (int i = 0; i < b->size(); i++) {
    offset_to_index(i, b->shape(), out_index);
    broadcast_index(out_index, b->shape(), a->shape(), in_index);
    int in_offset = index_to_offset(in_index, a->strides());
    b_data[i] = f(a_data[in_offset]);
  }
}

void TensorBackendCPU::zip(const std::shared_ptr<Tensor> &a,
                           const std::shared_ptr<Tensor> &b,
                           std::shared_ptr<Tensor> &c,
                           std::function<float(float, float)> f) {
  assert(a != nullptr && b != nullptr);
  // create tensor 'c' with the broadcasted shape of 'a' and 'b'
  std::vector<int> broadcasted_shape;
  assert(broadcast_shape(a->shape(), b->shape(), broadcasted_shape));
  if (c == nullptr) {
    c = a->zeros(broadcasted_shape);
  } else {
    assert(c->shape() == broadcasted_shape);
  }

  std::vector<int> a_index(a->shape().size(), 0);
  std::vector<int> b_index(b->shape().size(), 0);
  std::vector<int> c_index(c->shape().size(), 0);

  const float *a_data = a->data_->data_ptr();
  const float *b_data = b->data_->data_ptr();
  float *c_data = c->data_->mutable_data_ptr();

  for (int i = 0; i < c->size(); i++) {
    offset_to_index(i, c->shape(), c_index);
    broadcast_index(c_index, c->shape(), a->shape(), a_index);
    broadcast_index(c_index, c->shape(), b->shape(), b_index);
    int a_offset = index_to_offset(a_index, a->strides());
    int b_offset = index_to_offset(b_index, b->strides());
    c_data[i] = f(a_data[a_offset], b_data[b_offset]);
  }
}

void TensorBackendCPU::reduce(const std::shared_ptr<Tensor> &a,
                              std::shared_ptr<Tensor> &b, int dim,
                              std::function<float(float, float)> f) {
  // create tensor 'b' with the same shape as 'a', but the dim-th dimension is 1
  if (b == nullptr) {
    std::vector<int> shape(a->shape());
    shape[dim] = 1;
    b = a->zeros(shape);
  } else {
    auto tmp_shape = a->mutable_shape();
    tmp_shape[dim] = 1;
    assert(b->shape() == tmp_shape);
  }

  // DONE(lqb): convert dim to axis
  std::vector<int> out_index(b->shape().size(), 0);
  const float *a_data = a->data_->data_ptr();
  float *b_data = b->data_->mutable_data_ptr();

  for (int i = 0; i < b->size(); i++) {
    offset_to_index(i, b->shape(), out_index);
    for (int j = 0; j < a->shape()[dim]; j++) {
      out_index[dim] = j;
      int a_offset = index_to_offset(out_index, a->strides());
      b_data[i] = f(b_data[i], a_data[a_offset]);
    }
  }
}

/**
 * @brief get the offset of a tensor element given its index
 * @param index The index of the element
 * @param strides The strides of the tensor
 * @return The offset of the element
 */
int TensorBackendCPU::index_to_offset(const std::vector<int> &index,
                                      const std::vector<int> &strides) {
  assert(index.size() == strides.size());
  int offset = 0;
  for (int i = 0; i < index.size(); i++) {
    offset += index[i] * strides[i];
  }
  return offset;
}

/**
 * @brief get the index of a tensor element given its offset
 * @param offset The offset of the element
 * @param shape The shape of the tensor
 * @param index The index of the element
 * @return void
 */
void TensorBackendCPU::offset_to_index(int offset,
                                       const std::vector<int> &shape,
                                       std::vector<int> &index) {
  index.resize(shape.size());
  for (int i = shape.size() - 1; i >= 0; i--) {
    index[i] = offset % shape[i];
    offset /= shape[i];
  }
}

/**
 * @brief broadcast the index of a big tensor element to a smaller tensor
 * @param big_index The index of the element in the big tensor
 * @param big_shape The shape of the big tensor
 * @param shape The shape of the smaller tensor
 * @param broadcasted_index The broadcasted index
 * @return void
 */
void TensorBackendCPU::broadcast_index(const std::vector<int> &big_index,
                                       const std::vector<int> &big_shape,
                                       const std::vector<int> &shape,
                                       std::vector<int> &broadcasted_index) {
  broadcasted_index.resize(shape.size());
  // index 0 to offset is broadcasted
  int offset = big_shape.size() - shape.size();
  for (int i = 0; i < shape.size(); i++) {
    // if the shape is 1, means it is broadcasted
    broadcasted_index[i] = (shape[i] == 1 ? 0 : big_index[i + offset]);
  }
}

/**
 * @brief broadcast the shape of two tensors
 * @param shape1 The shape of the first tensor
 * @param shape2 The shape of the second tensor
 * @param broadcasted_shape The broadcasted shape
 * @return is broadcast success
 */
bool TensorBackendCPU::broadcast_shape(const std::vector<int> &shape1,
                                       const std::vector<int> &shape2,
                                       std::vector<int> &broadcasted_shape) {
  int n_dim = std::max(shape1.size(), shape2.size());
  broadcasted_shape.resize(n_dim);
  for (int i = 0; i < n_dim; i++) {
    int dim1 = i < shape1.size() ? shape1[i] : 1;
    int dim2 = i < shape2.size() ? shape2[i] : 1;
    if (!(dim1 == dim2 || dim1 == 1 || dim2 == 1)) {
      return false;
    }
    broadcasted_shape[i] = std::max(dim1, dim2);
  }
  return true;
}

/**
 * @brief negate a tensor element-wise
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Neg_Map(const std::shared_ptr<Tensor> &a,
                               std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return -x; });
}

/**
 * @brief element-wise inverse of a tensor
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Inv_Map(const std::shared_ptr<Tensor> &a,
                               std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return 1 / x; });
}

/**
 * @brief element-wise log of a tensor
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Log_Map(const std::shared_ptr<Tensor> &a,
                               std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return log(x); });
}

/**
 * @brief element-wise square root of a tensor
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Sqrt_Map(const std::shared_ptr<Tensor> &a,
                                std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return sqrt(x); });
}

/**
 * @brief element-wise power of a tensor
 * @param a The input tensor
 * @param b The power
 * @param c The result tensor
 */
void TensorBackendCPU::Pow_Map(const std::shared_ptr<Tensor> &a,
                               std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return pow(x, 2); });
}

/**
 * @brief element-wise exponential of a tensor
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Exp_Map(const std::shared_ptr<Tensor> &a,
                               std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return exp(x); });
}

/**
 * @brief element-wise RELU of a tensor
 * @note RELU(x) = max(0, x)
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Relu_Map(const std::shared_ptr<Tensor> &a,
                                std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) { return x > 0 ? x : 0; });
}

/**
 * @brief element-wise Sigmoid of a tensor
 * @note Sigmoid(x) = 1 / (1 + exp(-x))
 * @param a The input tensor
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Sigmoid_Map(const std::shared_ptr<Tensor> &a,
                                   std::shared_ptr<Tensor> &b) {
  map(a, b, [](float x) {
    if (x >= 0) {
      return 1 / (1 + exp(-x));
    } else {
      return exp(x) / (1 + exp(x));
    }
  });
}

/**
 * @brief add two tensors element-wise
 * @param a The first tensor
 * @param b The second tensor
 * @param c The result tensor
 * @return void
 */
void TensorBackendCPU::Add_Zip(const std::shared_ptr<Tensor> &a,
                               const std::shared_ptr<Tensor> &b,
                               std::shared_ptr<Tensor> &c) {
  zip(a, b, c, [](float x, float y) { return x + y; });
}

/**
 * @brief subtract two tensors element-wise
 * @param a The first tensor
 * @param b The second tensor
 * @param c The result tensor
 * @return void
 */
void TensorBackendCPU::Sub_Zip(const std::shared_ptr<Tensor> &a,
                               const std::shared_ptr<Tensor> &b,
                               std::shared_ptr<Tensor> &c) {
  zip(a, b, c, [](float x, float y) { return x - y; });
}

/**
 * @brief multiply two tensors element-wise
 * @param a The first tensor
 * @param b The second tensor
 * @param c The result tensor
 * @return void
 */
void TensorBackendCPU::Mul_Zip(const std::shared_ptr<Tensor> &a,
                               const std::shared_ptr<Tensor> &b,
                               std::shared_ptr<Tensor> &c) {
  zip(a, b, c, [](float x, float y) { return x * y; });
}

/**
 * @brief divide two tensors element-wise
 * @param a The first tensor
 * @param b The second tensor
 * @param c The result tensor
 * @return void
 */
void TensorBackendCPU::Div_Zip(const std::shared_ptr<Tensor> &a,
                               const std::shared_ptr<Tensor> &b,
                               std::shared_ptr<Tensor> &c) {
  zip(a, b, c, [](float x, float y) { return x / y; });
}

/**
 * @brief sum of all elements in a tensor along a dimension
 * @param a The input tensor
 * @param dim The dimension to sum along
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Sum_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                                  std::shared_ptr<Tensor> &b) {
  reduce(a, b, dim, [](float x, float y) { return x + y; });
}

/**
 * @brief mean of all elements in a tensor along a dimension
 * @param a The input tensor
 * @param dim The dimension to mean along
 * @param b The result tensor
 * @return void
 */
void TensorBackendCPU::Mean_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                                   std::shared_ptr<Tensor> &b) {
  // mean = sum / n
  reduce(a, b, dim, [](float x, float y) { return x + y; });
  int n = a->shape()[dim];
  reduce(b, b, dim, [n](float x, float y) { return x / n; });
}

/**
 * @brief multiply all elements in a tensor along a dimension
 * @param a The tensor to multiply
 * @param dim The dimension to multiply along，when dim = -1, multiply all
 * elements
 * @param b The result tensor
 */
void TensorBackendCPU::Mul_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                                  std::shared_ptr<Tensor> &b) {
  reduce(a, b, dim, [](float x, float y) { return x * y; });
}

/**
 * @brief matrix multiplication of two tensors
 *  Batched tensor matrix multiply ::
 *      for n:
 *          for i:
 *              for j:
 *                  for k:
 *                      out[n, i, j] += a[n, i, k] * b[n, k, j]
 * @param a The first tensor
 * @param b The second tensor
 * @param c The result tensor
 * @return void
 */
void TensorBackendCPU::MatMul(const std::shared_ptr<Tensor> &a,
                              const std::shared_ptr<Tensor> &b,
                              std::shared_ptr<Tensor> &c) {
  int a_batch_stride = a->shape()[0] > 1 ? a->strides()[0] : 0;
  int b_batch_stride = b->shape()[0] > 1 ? b->strides()[0] : 0;
  std::vector<int> a_stride_mod(a->strides().begin(), a->strides().end());
  std::vector<int> b_stride_mod(b->strides().begin(), b->strides().end());
  a_stride_mod[0] = a_batch_stride;
  b_stride_mod[0] = b_batch_stride;

  const float *a_data = a->data_->data_ptr();
  const float *b_data = b->data_->data_ptr();
  float *c_data = c->data_->mutable_data_ptr();

  for (int n = 0; n < c->shape()[0]; n++) {
    for (int i = 0; i < c->shape()[1]; i++) {
      for (int j = 0; j < c->shape()[2]; j++) {
        std::vector<int> c_index = {n, i, j};
        int out_offset = index_to_offset(c_index, c->strides());
        for (int k = 0; k < a->shape()[2]; k++) {
          std::vector<int> a_index = {n, i, k};
          std::vector<int> b_index = {n, k, j};
          int a_offset = index_to_offset(a_index, a_stride_mod);
          int b_offset = index_to_offset(b_index, b_stride_mod);
          c_data[out_offset] += a_data[a_offset] * b_data[b_offset];
        }
      }
    }
  }
}

// /**
//  * @brief transpose a tensor
//  * @param a The input tensor
//  * @param b The result tensor
//  * @return void
//  */
// void TensorBackendCPU::Transpose(const std::shared_ptr<Tensor> &a,
//                                  const std::shared_ptr<Tensor> &b) {
//   std::vector<int> in_index(a->shape().size(), 0);
//   std::vector<int> out_index(b->shape().size(), 0);
//   const float *a_data = a->data_->data_ptr();
//   float *b_data = b->data_->mutable_data_ptr();
// }

} // namespace runtime
} // namespace dlsys