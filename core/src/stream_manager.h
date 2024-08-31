/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-30 17:00:43
 * @Description  :
 */

#pragma once

#include "device_api_manager.h"
#include "dl_context.h"

namespace dlsys {
namespace runtime {

class StreamManager {
public:
  struct StreamLink {
    StreamLink(DLStreamHandle stream, StreamLink *parent = nullptr)
        : stream_(stream), parent_(parent) {}

    StreamLink(DLContext ctx, StreamLink *parent = nullptr) : parent_(parent) {
      DeviceAPIManager::Get(ctx)->StreamCreate(ctx, &stream_);
    }

    DLStreamHandle GetStream() { return stream_; }

    DLStreamHandle stream_;
    StreamLink *parent_;
  };

  class StreamGuard {
  public:
    StreamGuard(StreamLink **stream, DLContext ctx)
        : stream_ptr_(stream), ctx_(ctx) {
      old_stream_ = *stream;
      *stream_ptr_ = new StreamLink(ctx, old_stream_);
    }

    ~StreamGuard() {
      DeviceAPIManager::Get(ctx_)->StreamSync(ctx_, (*stream_ptr_)->stream_);
      DeviceAPIManager::Get(ctx_)->StreamDestroy(ctx_, (*stream_ptr_)->stream_);
      delete (*stream_ptr_);
      *stream_ptr_ = old_stream_;
      if (old_stream_->parent_ != nullptr) {
        old_stream_ = old_stream_->parent_;
      }
    }

    DLStreamHandle GetStream() { return (*stream_ptr_)->GetStream(); }

    StreamLink *old_stream_;
    StreamLink **stream_ptr_;
    DLContext ctx_;
  };

  StreamManager() = default;

  static StreamGuard NewStream(DLContext ctx) {
    return StreamGuard(&stream_link_, ctx);
  }

  static DLStreamHandle GetStream() { return stream_link_->GetStream(); }

  static void Init(DLContext ctx) {
    DeviceAPIManager::Get(ctx)->ContextCreate(ctx, &context_);
  }

  static void Destroy(DLContext ctx) {
    DeviceAPIManager::Get(ctx)->ContextDestroy(ctx, context_);
  }

  static DLContextHandle context_;
  static StreamLink *stream_link_;
};

} // namespace runtime
} // namespace dlsysls
