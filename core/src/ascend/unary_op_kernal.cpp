/*
 * @Author       : Li Qingbing(3263109808@qq.com)
 * @Version      : V0.0
 * @Date         : 2024-08-28 15:53:50
 * @Description  : 
 */

#include "custom_tiling.h"
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <typename srcType>
class UnaryOP {
public:
  __aicore__ inline UnaryOP() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              uint32_t totalLength, uint32_t tileNum, NPU_OP_TYPE op_type) {
    this->blockLength = totalLength / GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
    this->op_type_ = op_type;
    xGm.SetGlobalBuffer((__gm__ srcType *)x + this->blockLength * GetBlockIdx(),
                        this->blockLength);
    yGm.SetGlobalBuffer((__gm__ srcType *)y + this->blockLength * GetBlockIdx(),
                        this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(srcType));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(srcType));
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
    DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    LocalTensor<srcType> xLocal = inQueueX.DeQue<srcType>();
    LocalTensor<srcType> yLocal = outQueueY.AllocTensor<srcType>();
    switch (this->op_type_) {
    case NPU_OP_TYPE::EXP:
      Exp(yLocal, xLocal, this->tileLength);
      break;
    case NPU_OP_TYPE::LN:
      Ln(yLocal, xLocal, this->tileLength);
      break;
    case NPU_OP_TYPE::SQRT:
      Sqrt(yLocal, xLocal, this->tileLength);
      break;
    case NPU_OP_TYPE::RELU:
      Relu(yLocal, xLocal, this->tileLength);
      break;
    }
    outQueueY.EnQue<srcType>(yLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<srcType> yLocal = outQueueY.DeQue<srcType>();
    DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
    outQueueY.FreeTensor(yLocal);
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  GlobalTensor<srcType> xGm;
  GlobalTensor<srcType> yGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  NPU_OP_TYPE op_type_;
};

extern "C" __global__ __aicore__ void unary_op_kernal(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, CustomTilingData tiling)
{
  // printf("add_custom, totalLength=%d, tileNum=%d\n", tiling.totalLength,
  //        tiling.tileNum);
  UnaryOP<half> op;
  op.Init(x, y, tiling.totalLength, tiling.tileNum, static_cast<NPU_OP_TYPE>(tiling.opType));
  op.Process();
}


