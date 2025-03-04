/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-18 22:49:38
 * @Description  : 
 */

#include "dl_context.h"
#include "tensor.h"
#include "tensor_helper.h"
#include "stream_manager.h"
#include <iostream>
#include <vector>

using namespace dlsys::runtime;

int main() {
  std::cout << "***********************************" << std::endl;
  DLContext ctx = {0, DLDeviceType::KAtlas};
  StreamManager::Init(ctx);
  auto stream = StreamManager::NewStream(ctx);
  // t1 = {1.0, 2.0, 3.0}
  auto t1 = dlsys::runtime::tensor({0, DLDeviceType::kCPU}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  t1->set_requires_grad(true);
  std::cout << t1->to_string() << std::endl;
  // t2 = {4.0, 5.0, 6.0}
  auto t2 = dlsys::runtime::tensor({0, DLDeviceType::kCPU}, std::vector<float>{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0});
  t2->set_requires_grad(true);
  std::cout << t2->to_string() << std::endl;

  // t3 = {5.0, 7.0, 9.0}
  auto t3 = t1 + t2;
  // t4 = {5.0, 14.0, 27.0}
  auto t4 = t3 * t1;

  // t3->to({0, DLDeviceType::kCPU});
  // t4->to({0, DLDeviceType::kCPU});

  std::cout << t3->to_string() << std::endl;
  std::cout << t4->to_string() << std::endl;

  // std::cout << "***********************************" << std::endl;
  // std::cout << t1->to_string() << std::endl;
  // std::cout << t2->to_string() << std::endl;
  // std::cout << t3->to_string() << std::endl;
  // std::cout << t4->to_string() << std::endl;
  // std::cout << "***********************************" << std::endl;

  std::cout << "****************** backward *****************" << std::endl;

  t4->backward(dlsys::runtime::tensor({0, DLDeviceType::kCPU}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
  // grad = {6.0, 18.0, 36.0}
  std::cout << t1->to_string() << std::endl;
  // grad = {1.0, 4.0, 9.0}
  std::cout << t2->to_string() << std::endl;
  // grad = {1.0, 4.0, 9.0}
  std::cout << t3->to_string() << std::endl;
  // grad = {1.0, 2.0, 3.0}
  std::cout << t4->to_string() << std::endl;

  // StreamManager::Destroy(ctx);
  return 0;
}