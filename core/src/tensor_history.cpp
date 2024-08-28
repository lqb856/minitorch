/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-19 01:20:59
 * @Description  :
 */

#include "tensor_data.h"

#include "tensor.h"
#include "tensor_history.h"

namespace dlsys {
namespace runtime {

std::string TensorHistory::to_string() const {
  std::string ret = "TensorHistory(";
  ret += "inputs=[";
  for (size_t i = 0; i < inputs_.size(); ++i) {
    ret += inputs_[i]->name();
    if (i != inputs_.size() - 1)
      ret += ", ";
  }
  ret += "], ";
  ret += "last_function=";
  ret += (last_function_.has_value() ? last_function_.value()->func_name() : "None");
  ret += ")";
  return ret;
}

} // namespace runtime
} // namespace dlsys