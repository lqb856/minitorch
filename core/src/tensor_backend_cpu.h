/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-29 16:39:27
 * @Description  : CPU beackend
 */

#pragma once

#include "tensor_backend.h"
#include <functional>

namespace dlsys {
namespace runtime {

class TensorBackendCPU : public TensorBackend {
public:
  ~TensorBackendCPU(){};

  /**
   * @brief get the offset of a tensor element given its index
   * @param index The index of the element
   * @param strides The strides of the tensor
   * @return The offset of the element
   */
  int index_to_offset(const std::vector<int> &index,
                      const std::vector<int> &strides);

  /**
   * @brief get the index of a tensor element given its offset
   * @param offset The offset of the element
   * @param shape The shape of the tensor
   * @param index The index of the element
   * @return void
   */
  void offset_to_index(int offset, const std::vector<int> &shape,
                       std::vector<int> &index);

  /**
   * @brief broadcast the index of a big tensor element to a smaller tensor
   * @param big_index The index of the element in the big tensor
   * @param big_shape The shape of the big tensor
   * @param shape The shape of the smaller tensor
   * @param broadcasted_index The broadcasted index
   * @return void
   */
  void broadcast_index(const std::vector<int> &big_index,
                       const std::vector<int> &big_shape,
                       const std::vector<int> &shape,
                       std::vector<int> &broadcasted_index);

  /**
   * @brief broadcast the shape of two tensors
   * @param shape1 The shape of the first tensor
   * @param shape2 The shape of the second tensor
   * @param broadcasted_shape The broadcasted shape
   * @return is broadcast success
   */
  bool broadcast_shape(const std::vector<int> &shape1,
                       const std::vector<int> &shape2,
                       std::vector<int> &broadcasted_shape);

  /**
   * @brief negate a tensor element-wise
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Neg_Map(const std::shared_ptr<Tensor> &a,
               std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise inverse of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Inv_Map(const std::shared_ptr<Tensor> &a,
               std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise log of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Log_Map(const std::shared_ptr<Tensor> &a,
               std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise square root of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Sqrt_Map(const std::shared_ptr<Tensor> &a,
                std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise power of a tensor
   * @param a The input tensor
   * @param b The power
   * @param c The result tensor
   */
  void Pow_Map(const std::shared_ptr<Tensor> &a,
               std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise exponential of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Exp_Map(const std::shared_ptr<Tensor> &a,
               std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise RELU of a tensor
   * @note RELU(x) = max(0, x)
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Relu_Map(const std::shared_ptr<Tensor> &a,
                std::shared_ptr<Tensor> &b) override;

  /**
   * @brief element-wise Sigmoid of a tensor
   * @note Sigmoid(x) = 1 / (1 + exp(-x))
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  void Sigmoid_Map(const std::shared_ptr<Tensor> &a,
                   std::shared_ptr<Tensor> &b) override;

  /**
   * @brief add two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  void Add_Zip(const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b,
               std::shared_ptr<Tensor> &c) override;

  /**
   * @brief subtract two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  void Sub_Zip(const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b,
               std::shared_ptr<Tensor> &c) override;

  /**
   * @brief multiply two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  void Mul_Zip(const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b,
               std::shared_ptr<Tensor> &c) override;

  /**
   * @brief divide two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  void Div_Zip(const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b,
               std::shared_ptr<Tensor> &c) override;

  /**
   * @brief sum of all elements in a tensor along a dimension
   * @param a The input tensor
   * @param dim The dimension to sum along
   * @param b The result tensor
   * @return void
   */
  void Sum_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                  std::shared_ptr<Tensor> &b) override;

  /**
   * @brief mean of all elements in a tensor along a dimension
   * @param a The input tensor
   * @param dim The dimension to mean along
   * @param b The result tensor
   * @return void
   */
  void Mean_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                   std::shared_ptr<Tensor> &b) override;

  /**
   * @brief multiply all elements in a tensor along a dimension
   * @param a The tensor to multiply
   * @param dim The dimension to multiply along，when dim = -1, multiply all
   * elements
   * @param b The result tensor
   */
  void Mul_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                  std::shared_ptr<Tensor> &b) override;

  /**
   * @brief batched matrix multiplication (n, m, k) @ (n, k, p) -> (n, m, p)
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  void MatMul(const std::shared_ptr<Tensor> &a,
              const std::shared_ptr<Tensor> &b,
              std::shared_ptr<Tensor> &c) override;

  // /**
  //  * @brief transpose a tensor
  //  * @param a The input tensor
  //  * @param b The result tensor
  //  * @return void
  //  */
  // void Transpose(const std::shared_ptr<Tensor> &a,
  //                const std::shared_ptr<Tensor> &b) override;

private:
  void map(const std::shared_ptr<Tensor> &a, std::shared_ptr<Tensor> &b,
           std::function<float(float)> f);
  void zip(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
           std::shared_ptr<Tensor> &c,
           std::function<float(float, float)> f);
  void reduce(const std::shared_ptr<Tensor> &a,
              std::shared_ptr<Tensor> &b, int dim,
              std::function<float(float, float)> f);
};

} // namespace runtime
} // namespace dlsys