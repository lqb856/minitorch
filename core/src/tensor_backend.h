/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-29 16:38:40
 * @Description  :
 */

#pragma once

// #include "tensor.h"
#include <memory>
#include <vector>

namespace dlsys {
namespace runtime {

class Tensor;

class TensorBackend {
public:
  virtual ~TensorBackend(){};

  // /**
  //  * @brief get the offset of a tensor element given its index
  //  * @param index The index of the element
  //  * @param strides The strides of the tensor
  //  * @return The offset of the element
  //  */
  // virtual int index_to_offset(const std::vector<int> &index,
  //                             const std::vector<int> &strides) = 0;

  // /**
  //  * @brief get the index of a tensor element given its offset
  //  * @param offset The offset of the element
  //  * @param shape The shape of the tensor
  //  * @param index The index of the element
  //  * @return void
  //  */
  // virtual void offset_to_index(int offset, const std::vector<int> &shape,
  //                              std::vector<int> &index) = 0;

  // /**
  //  * @brief broadcast the index of a big tensor element to a smaller tensor
  //  * @param big_index The index of the element in the big tensor
  //  * @param big_shape The shape of the big tensor
  //  * @param shape The shape of the smaller tensor
  //  * @param broadcasted_index The broadcasted index
  //  * @return void
  //  */
  // virtual void broadcast_index(const std::vector<int> &big_index,
  //                              const std::vector<int> &big_shape,
  //                              const std::vector<int> &shape,
  //                              std::vector<int> &broadcasted_index) = 0;

  // /**
  //  * @brief broadcast the shape of two tensors
  //  * @param shape1 The shape of the first tensor
  //  * @param shape2 The shape of the second tensor
  //  * @param broadcasted_shape The broadcasted shape
  //  * @return is broadcast success
  //  */
  // virtual bool broadcast_shape(const std::vector<int> &shape1,
  //                              const std::vector<int> &shape2,
  //                              std::vector<int> &broadcasted_shape) = 0;
  /**
   * @brief negate a tensor element-wise
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Neg_Map(const std::shared_ptr<Tensor> &a,
                       std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise inverse of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Inv_Map(const std::shared_ptr<Tensor> &a,
                       std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise log of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Log_Map(const std::shared_ptr<Tensor> &a,
                       std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise square root of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Sqrt_Map(const std::shared_ptr<Tensor> &a,
                        std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise power of a tensor
   * @param a The input tensor
   * @param b The power
   * @param c The result tensor
   */
  virtual void Pow_Map(const std::shared_ptr<Tensor> &a,
                       std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise exponential of a tensor
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Exp_Map(const std::shared_ptr<Tensor> &a,
                       std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise RELU of a tensor
   * @note RELU(x) = max(0, x)
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Relu_Map(const std::shared_ptr<Tensor> &a,
                        std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief element-wise Sigmoid of a tensor
   * @note Sigmoid(x) = 1 / (1 + exp(-x))
   * @param a The input tensor
   * @param b The result tensor
   * @return void
   */
  virtual void Sigmoid_Map(const std::shared_ptr<Tensor> &a,
                           std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief add two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  virtual void Add_Zip(const std::shared_ptr<Tensor> &a,
                       const std::shared_ptr<Tensor> &b,
                       std::shared_ptr<Tensor> &c) = 0;

  /**
   * @brief subtract two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  virtual void Sub_Zip(const std::shared_ptr<Tensor> &a,
                       const std::shared_ptr<Tensor> &b,
                       std::shared_ptr<Tensor> &c) = 0;

  /**
   * @brief multiply two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  virtual void Mul_Zip(const std::shared_ptr<Tensor> &a,
                       const std::shared_ptr<Tensor> &b,
                       std::shared_ptr<Tensor> &c) = 0;

  /**
   * @brief divide two tensors element-wise
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  virtual void Div_Zip(const std::shared_ptr<Tensor> &a,
                       const std::shared_ptr<Tensor> &b,
                       std::shared_ptr<Tensor> &c) = 0;

  /**
   * @brief sum of all elements in a tensor along a dimension
   * @param a The input tensor
   * @param dim The dimension to sum along
   * @param b The result tensor
   * @return void
   */
  virtual void Sum_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                          std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief mean of all elements in a tensor along a dimension
   * @param a The input tensor
   * @param dim The dimension to mean along
   * @param b The result tensor
   * @return void
   */
  virtual void Mean_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                           std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief multiply all elements in a tensor along a dimension
   * @param a The tensor to multiply
   * @param dim The dimension to multiply along，when dim = -1, multiply all
   * elements
   * @param b The result tensor
   */
  virtual void Mul_Reduce(const std::shared_ptr<Tensor> &a, int dim,
                          std::shared_ptr<Tensor> &b) = 0;

  /**
   * @brief matrix multiplication of two tensors
   * @param a The first tensor
   * @param b The second tensor
   * @param c The result tensor
   * @return void
   */
  virtual void MatMul(const std::shared_ptr<Tensor> &a,
                      const std::shared_ptr<Tensor> &b,
                      std::shared_ptr<Tensor> &c) = 0;

  // /**
  //  * @brief transpose a tensor
  //  * @param a The input tensor
  //  * @param b The result tensor
  //  * @return void
  //  */
  // virtual void Transpose(const std::shared_ptr<Tensor> &a,
  //                        const std::shared_ptr<Tensor> &b) = 0;

  // virtual void Softmax(const Tensor& a, Tensor& b) = 0;
  // virtual void Conv2D(const Tensor& a, const Tensor& b, Tensor& c) = 0;
  // virtual void Pool2D(const Tensor& a, Tensor& b) = 0;
  // virtual void Dropout(const Tensor& a, Tensor& b) = 0;
  // virtual void BatchNorm(const Tensor& a, Tensor& b) = 0;
};

} // namespace runtime
} // namespace dlsys