/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-28 15:54:09
 * @Description  : 
 */

#pragma once

#include <cstdint>
#include <stdint.h>

enum class NPU_OP_TYPE {
  ADD = 0,
  SUB,
  MUL,
  DIV,
  MAX,
  MIN,
  POW,
  SQRT,
  EXP,
  LN,
  SIGMOD,
  RELU,
};

struct CustomTilingData {
  uint32_t opType;
  uint32_t totalLength;
  uint32_t tileNum;
  uint32_t blockLength;
};