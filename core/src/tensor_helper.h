/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-11 19:07:20
 * @Description  : 
 */

#pragma once

#include "tensor.h"
#include <iostream>
#include <type_traits>

namespace dlsys {
namespace runtime {

/**
 * @brief get the element type of a container
 * @tparam T container type
 */
template <typename T> struct ElementType {
  using type = typename T::value_type;
};

/**
 * @brief check if a type is a vector
 */
template <typename T> struct IsVector : std::false_type {};
template <typename T> struct IsVector<std::vector<T>> : std::true_type {};

/**
 * @brief debug helper func to print the type of a container
 * @tparam T container type
 */
template <typename T> void printType() {
  if constexpr (IsVector<T>::value) {
    using InnerType = typename T::value_type;
    std::cout << "std::vector<";
    printType<InnerType>();
    std::cout << ">";
  } else {
    std::cout << "element_type";
  }
}

/**
 * @brief get the shape of a vector recursively
 * @tparam T vector type
 * @param vec vector
 * @param shape shape of the vector
 */
template <typename T>
void getShapeRecursive(const T &vec, std::vector<int> &shape) {
  if (vec.empty())
    return;

  shape.push_back(vec.size());
  using InnerType = typename ElementType<
      typename std::remove_reference<decltype(vec)>::type>::type;

  if constexpr (IsVector<InnerType>::value) {
    getShapeRecursive(vec[0], shape);
  }
}

/**
 * @brief get the shape of a vector
 * @tparam T vector type
 * @param vec vector
 * @return shape of the vector
 */
template <typename T> std::vector<int> getShape(const T &vec) {
  static_assert(IsVector<T>::value, "T must be a std::vector");
  std::vector<int> shape;
  getShapeRecursive(vec, shape);
  return shape;
}

/**
 * @brief get the deepest element type of a container
 * @tparam T container type
 */
template <typename T> struct GetElementType {
  using type = typename T::value_type;
};
template <typename T> struct GetElementType<std::vector<T>> {
  using type = typename GetElementType<T>::type;
};
template <> struct GetElementType<int> {
  using type = int;
};
template <> struct GetElementType<float> {
  using type = float;
};
template <> struct GetElementType<double> {
  using type = double;
};

/**
 * @brief flatten a multi-dimensional std::vector
 * @tparam T vector type
 * @param vec vector
 * @return flattened vector
 */
template <typename T>
void flattenRecursive(const T &vec,
                      std::vector<typename GetElementType<T>::type> &result) {
  for (const auto &element : vec) {
    using InnerType = typename ElementType<
        typename std::remove_reference<decltype(vec)>::type>::type;
    if constexpr (std::is_same<typename GetElementType<T>::type,
                               InnerType>::value) {
      // if the element is the deepest type, add it to the result
      result.push_back(element);
    } else {
      // if the element is a vector, recursively process it
      flattenRecursive(element, result);
    }
  }
}

/**
 * @brief flatten a multi-dimensional std::vector
 * @tparam T vector type
 * @param vec vector
 * @return flattened vector
 */
template <typename T>
std::vector<typename GetElementType<T>::type> flatten(const T &vec) {
  std::vector<typename GetElementType<T>::type> result;
  flattenRecursive(vec, result);
  return result;
}

/**
 * @brief create a tensor with given data and context
 * @param data data of the tensor, must be a vector of float, could be any dimension
 * @param ctx context of the tensor, e.g. CPU or GPU, context must coodinate with the data position
 * @return tensor
 */
template <typename T>
std::shared_ptr<Tensor> tensor(DLContext ctx, const T &data) {
  auto shape = getShape(data);
  auto flattened_data = flatten(data);
  auto tensor = Tensor::make(ctx, shape, flattened_data.data());
  return tensor;
}

} // namespace runtime
} // namespace dlsys