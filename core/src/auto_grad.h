/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-17 16:00:04
 * @Description  : 
 */

#pragma once

#include <future>
#include <memory>
#include <vector>

namespace dlsys {
namespace runtime {

class Tensor;

std::vector<std::shared_ptr<Tensor>> topological_sort(const std::shared_ptr<Tensor> &input);

void back_propagate(const std::shared_ptr<Tensor> &intput, const std::shared_ptr<Tensor> &grad);

} // namespace runtime
} // namespace dlsys