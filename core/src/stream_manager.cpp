/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-30 23:17:23
 * @Description  : 
 */

#include "stream_manager.h"
#include "dl_context.h"

namespace dlsys {
namespace runtime {

StreamManager::StreamLink *StreamManager::stream_link_ = nullptr;

DLContextHandle StreamManager::context_ = nullptr;

} // namespace runtime
} // namespace dlsys