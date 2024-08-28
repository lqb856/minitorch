#include <iostream>
#include <type_traits>
#include <vector>

// 检查是否为 std::vector
template <typename T> struct IsVector : std::false_type {};
template <typename T> struct IsVector<std::vector<T>> : std::true_type {};

// 递归获取嵌套层次
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

// 获取 std::vector 的元素类型
template <typename T> struct ElementType {
  using type = typename T::value_type;
};

// 基础递归模板，用于处理任意维度的 std::vector
template <typename T>
void getShapeRecursive(const T &vec, std::vector<size_t> &shape) {
  if (vec.empty())
    return;

  shape.push_back(vec.size());
  // printType<T>();

  using InnerType = typename ElementType<
      typename std::remove_reference<decltype(vec)>::type>::type;

  if constexpr (IsVector<InnerType>::value) {
    getShapeRecursive(vec[0], shape);
  }
}

// 主函数用于获取 shape
template <typename T> std::vector<size_t> getShape(const T &vec) {
  static_assert(IsVector<T>::value, "T must be a std::vector");
  std::vector<size_t> shape;
  getShapeRecursive(vec, shape);
  return shape;
}

// 辅助结构体，用于提取最深层次的元素类型
template <typename T> struct GetElementType {
  using type = typename T::value_type;
};

template <typename T> struct GetElementType<std::vector<T>> {
  using type = typename GetElementType<T>::type;
};

template <> struct GetElementType<int> {
  using type = int;
};

// 递归展开任意层次的 std::vector
template <typename T>
void flattenRecursive(const T &vec,
                      std::vector<typename GetElementType<T>::type> &result) {
  for (const auto &element : vec) {
    using InnerType = typename ElementType<
        typename std::remove_reference<decltype(vec)>::type>::type;
    if constexpr (std::is_same<typename GetElementType<T>::type,
                               InnerType>::value) {
      // 如果元素是最深层次的类型，添加到结果中
      result.push_back(element);
    } else {
      // 如果元素是向量，递归处理
      flattenRecursive(element, result);
    }
  }
}

// 展开多维 std::vector
template <typename T>
std::vector<typename GetElementType<T>::type> flatten(const T &vec) {
  std::vector<typename GetElementType<T>::type> result;
  flattenRecursive(vec, result);
  return result;
}

int main() {
  // Test cases

  std::vector<std::vector<int>> v1 = {{1}, {2}, {3}};
  std::vector<size_t> shape1 = getShape(v1);
  std::cout << "Shape of v1: { ";
  for (size_t s : shape1)
    std::cout << s << " ";
  std::cout << "}" << std::endl;

  std::vector<std::vector<int>> v2 = {{1, 2}, {3, 4}, {5, 6}};
  std::vector<size_t> shape2 = getShape(v2);
  std::cout << "Shape of v2: { ";
  for (size_t s : shape2)
    std::cout << s << " ";
  std::cout << "}" << std::endl;

  std::vector<std::vector<std::vector<int>>> v3 = {
      {{1}, {2}}, {{3}, {4}}, {{3}, {4}}};
  std::vector<size_t> shape3 = getShape(v3);
  std::cout << "Shape of v3: { ";
  for (size_t s : shape3)
    std::cout << s << " ";
  std::cout << "}" << std::endl;

  auto flat1 = flatten(v1);
  std::cout << "Flattened v1: { ";
  for (int num : flat1)
    std::cout << num << " ";
  std::cout << "}" << std::endl;

  auto flat2 = flatten(v2);
  std::cout << "Flattened v2: { ";
  for (int num : flat2)
    std::cout << num << " ";
  std::cout << "}" << std::endl;

  auto flat3 = flatten(v3);
  std::cout << "Flattened v3: { ";
  for (int num : flat3)
    std::cout << num << " ";
  std::cout << "}" << std::endl;
  return 0;
}
