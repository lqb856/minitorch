/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-17 15:59:43
 * @Description  :
 */

#include "auto_grad.h"
#include "tensor.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace dlsys {
namespace runtime {

// FIXME: the implementation of topological_sort is not correct
void dfs(const std::shared_ptr<Tensor> &node, std::unordered_set<int> &marked,
         std::unordered_set<int> &path,
         std::vector<std::shared_ptr<Tensor>> &result) {
  if (node == nullptr || node->is_constant()) {
    return;
  }
  if (marked.find(node->unique_id_) != marked.end()) {
    return;
  }
  if (path.find(node->unique_id_) != path.end()) {
    throw std::runtime_error("Graph has cycle");
  }
  path.insert(node->unique_id_);
  auto children = node->parents();
  for (auto &child : children) {
    dfs(child, marked, path, result);
  }
  path.erase(node->unique_id_);
  marked.insert(node->unique_id_);
  result.push_back(node);
}

std::vector<std::shared_ptr<Tensor>>
topological_sort(const std::shared_ptr<Tensor> &input) {
  std::vector<std::shared_ptr<Tensor>> result;
  std::unordered_set<int> marked;
  std::unordered_set<int> path;
  dfs(input, marked, path, result);
  std::reverse(result.begin(), result.end());
  return result;
}


void back_propagate(const std::shared_ptr<Tensor> &intput,
                    const std::shared_ptr<Tensor> &grad) {
  std::unordered_map<int, std::shared_ptr<Tensor>> grad_map;
  auto nodes = topological_sort(intput);
  grad_map[intput->unique_id_] = grad;
  std::shared_ptr<Tensor> cur_grad;

  // print debug info
  for (auto &node : nodes) {
    std::cout << "node: " << node->to_string() << std::endl;
  }

  for (auto &node : nodes) {
    if (grad_map.find(node->unique_id_) != grad_map.end()) {
      cur_grad = grad_map[node->unique_id_];
    }
    // TODO(lqb): only accumulate grad for leaf node?
    // what if the node is not leaf node and it requires grad?
    if (node->is_leaf()) {
      node->accumulate_grad(cur_grad);
      continue;
    }
    auto out_grad = node->chain_rule(cur_grad);
    auto parents = node->parents();
    for (int i = 0; i < parents.size(); i++) {
      auto iter = grad_map.find(parents[i]->unique_id_);
      if (iter == grad_map.end()) {
        grad_map[parents[i]->unique_id_] = out_grad[i];
      } else {
        iter->second = iter->second + out_grad[i];
      }
    }
  }
}

} // namespace runtime
} // namespace dlsys