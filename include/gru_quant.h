#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "quantize_ops_helper.hpp"

namespace gru {

// 模板参数说明:
// - XT: 输入 x 的量化类型
// - HT: 隐藏状态 h 的量化类型
// - WT: 权重矩阵 W 的量化类型
// - RT: 递归权重矩阵 R 的量化类型
template <typename XT, typename HT, typename WT, typename RT>
class ForwardPassQuant {
   public:
    // training: `true` if the caller intends to perform a backward pass to compute gradients.
    // batch_size: the number of training/inference inputs provided in each tensor.
    // input_size: the dimension of each input vector.
    // hidden_size: the expected dimension of each output vector.
    // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
    ForwardPassQuant(const bool training, const int batch_size, const int input_size,
                     const int hidden_size, const cublasHandle_t &blas_handle,
                     const cudaStream_t &stream = 0);

    // Releases internal resources.
    // Blocks until all iterations have completed executing on the GPU.
    ~ForwardPassQuant();

    void setRescaleParam(const GRUQuantitativeParameters &parms);

    // 简化的 GRU 前向接口（内部管理临时缓冲区）
    //
    // W: [C,H*3] 输入权重矩阵（量化后）
    // R: [H,H*3] 循环权重矩阵（量化后）
    // bx: [H*3] 输入偏置（量化后）
    // br: [H*3] 循环偏置（量化后）
    // x: [N,C] 输入序列
    // h: [N,H] 初始隐藏状态（输入）和输出隐藏状态
    // v: [N,H*4] 中间激活值（训练模式需要）
    // zoneout_prob: Zoneout 概率
    // zoneout_mask: [N,H] Zoneout mask
    void Run(const int steps, const WT *W, const RT *R, const int32_t *bx,
             const int32_t *br, const XT *x, HT *h, int32_t *v,
             const float zoneout_prob, const HT *zoneout_mask);

   private:
    // 内部迭代函数
    // cur_Wx_: 当前时间步的 W @ x 结果（指向 tmp_Wx_ 的偏移）
    void IterateInternal(const RT *R, const int32_t *bx, const int32_t *br, const HT *h,
                         HT *h_out, int32_t *v, const int32_t *cur_Wx_,
                         const float zoneout_prob, const HT *zoneout_mask);

    // 计算 W @ x GEMM 并 rescale（输出到 tmp_Wx_）
    void ComputeWx(const WT *W, const XT *x, int steps);

    // 计算 R @ h GEMM 并 rescale（输出到 tmp_Rh_）
    void ComputeRh(const RT *R, const HT *h);

    // 预分配内存缓冲区
    void EnsureBuffersAllocated(int steps);

    // 预计算权重相关的常量
    void PrecomputeWeightSums(const WT *W, const RT *R);

    struct private_data;
    private_data *data_;

    QuantGRUReScale rescale_param_;

    // 预分配的内部缓冲区（使用 dev::vector 自动管理内存）
    int max_steps_ = 0;

    // GEMM 中间结果（int64 避免溢出）
    dev::vector<int64_t> tmp_Wx_i64_;   // [hidden*3 * max_steps * batch]
    dev::vector<int64_t> tmp_Rh_i64_;   // [hidden*3 * batch]

    // GEMM rescale 后的结果（int32 供 gate 计算使用）
    dev::vector<int32_t> tmp_Wx_;       // [hidden*3 * max_steps * batch]
    dev::vector<int32_t> tmp_Rh_;       // [hidden*3 * batch]

    // 权重和常量（预计算）
    dev::vector<int64_t> W_sum_mul_x_zp_;  // [hidden*3]
    dev::vector<int64_t> R_sum_mul_h_zp_;  // [hidden*3]
    bool weight_sums_computed_ = false;

    // 缓存的权重指针（用于检测权重是否变化）
    const WT *cached_W_ = nullptr;
    const RT *cached_R_ = nullptr;
};

}  // namespace gru
