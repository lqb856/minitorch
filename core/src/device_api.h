/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-07-29 15:11:44
 * @Description  :
 */

#pragma once

#include <assert.h>
#include <string>
#include "dl_context.h"

namespace dlsys {
namespace runtime {

class DeviceAPI {
public:
  virtual ~DeviceAPI() {}

  /**
   * @brief Allocate a data space on the device.
   * @param ctx The context of the device.
   * @param size The size of the memory to be allocated.
   * @param alignment The alignment of the memory.
   * @return The allocated memory.
   */
  virtual void *AllocDataSpace(DLContext ctx, size_t size,
                               size_t alignment) = 0;

  /**
   * @brief Free a data space on the device.
   * @param ctx The context of the device.
   * @param ptr The pointer to the memory to be deallocated.
   */
  virtual void FreeDataSpace(DLContext ctx, void *ptr) = 0;

  /**
   * @brief Copy data between two address.
   * @param from The source address.
   * @param to The target address.
   * @param size The size of the memory to be copied.
   * @param ctx_from The context of the source device.
   * @param ctx_to The context of the target device.
   * @param stream The stream where the copy is executed, can be NULL.
   */
  virtual void CopyDataFromTo(const void *from, void *to, size_t size,
                              DLContext ctx_from, DLContext ctx_to,
                              DLStreamHandle stream) = 0;


  /**
   * @brief Create a stream for copy and compute.
   * @param ctx The context of the device.
   * @return The created stream.
   */
  virtual void StreamCreate(DLContext ctx, DLStreamHandle *stream) = 0;

  /**
   * @brief Destroy a created stream.
   * @param ctx The context of the device.
   * @param stream The stream to be destroyed.
   */
  virtual void StreamDestroy(DLContext ctx, DLStreamHandle stream) = 0;

  /**
   * @brief Create a stream for copy and compute.
   * @param ctx The context of the device.
   * @return The created stream.
   */
  virtual void StreamSync(DLContext ctx, DLStreamHandle stream) = 0;

  /**
   * @brief Create a context for the device.
   * @param ctx The context of the device.
   */
  virtual void ContextCreate(DLContext ctx, DLContextHandle *context) = 0;

  /**
   * @brief Destroy a created context.
   * @param ctx The context of the device.
   * @param context The context to be destroyed.
   */
  virtual void ContextDestroy(DLContext ctx, DLContextHandle context) = 0;
};

} // namespace runtime
} // namespace dlsys