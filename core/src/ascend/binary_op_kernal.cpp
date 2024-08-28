/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend
 * plaform.
 */

#include "custom_tiling.h"
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <typename srcType>
class BinaryOP {
public:
  __aicore__ inline BinaryOP() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t totalLength, uint32_t tileNum,
                              NPU_OP_TYPE op_type) {
    this->blockLength = totalLength / GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
    this->op_type_ = op_type;
    xGm.SetGlobalBuffer((__gm__ srcType *)x + this->blockLength * GetBlockIdx(),
                        this->blockLength);
    yGm.SetGlobalBuffer((__gm__ srcType *)y + this->blockLength * GetBlockIdx(),
                        this->blockLength);
    zGm.SetGlobalBuffer((__gm__ srcType *)z + this->blockLength * GetBlockIdx(),
                        this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(srcType));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(srcType));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(srcType));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

private:
  __aicore__ inline void CopyIn(int32_t progress) {
    LocalTensor<srcType> xLocal = inQueueX.AllocTensor<srcType>();
    LocalTensor<srcType> yLocal = inQueueY.AllocTensor<srcType>();
    DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    LocalTensor<srcType> xLocal = inQueueX.DeQue<srcType>();
    LocalTensor<srcType> yLocal = inQueueY.DeQue<srcType>();
    LocalTensor<srcType> zLocal = outQueueZ.AllocTensor<srcType>();
    switch (this->op_type_) {
    case NPU_OP_TYPE::ADD:
      Add(zLocal, xLocal, yLocal, this->tileLength);
      break;
    case NPU_OP_TYPE::SUB:
      Sub(zLocal, xLocal, yLocal, this->tileLength);
      break;
    case NPU_OP_TYPE::MUL:
      Mul(zLocal, xLocal, yLocal, this->tileLength);
      break;
    case NPU_OP_TYPE::DIV:
      Div(zLocal, xLocal, yLocal, this->tileLength);
      break;
    }
    // Add(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue<srcType>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<srcType> zLocal = outQueueZ.DeQue<srcType>();
    DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  GlobalTensor<srcType> xGm;
  GlobalTensor<srcType> yGm;
  GlobalTensor<srcType> zGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  NPU_OP_TYPE op_type_;
};

extern "C" __global__ __aicore__ void binary_op(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                GM_ADDR workspace,
                                                CustomTilingData tiling) {
  // printf("add_custom, totalLength=%d, tileNum=%d\n", tiling.totalLength,
  //        tiling.tileNum);
  BinaryOP<half> op;
  op.Init(x, y, z, tiling.totalLength, tiling.tileNum,
          static_cast<NPU_OP_TYPE>(tiling.opType));
  op.Process();
}
