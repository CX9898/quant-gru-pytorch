#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "quantize_ops_helper.h"

namespace gru {

// 量化 GRU 前向传播类
// 所有量化值统一使用 int32_t 存储，实际值通过 clamp_by_bitwidth 限制到对应位宽
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
    // 所有输入输出数据使用 int32_t 存储，实际值限制在配置的位宽范围内
    //
    // W: [C,H*3] 输入权重矩阵（量化后，int32_t 存储）
    // R: [H,H*3] 循环权重矩阵（量化后，int32_t 存储）
    // bx: [H*3] 输入偏置（量化后）
    // br: [H*3] 循环偏置（量化后）
    // x: [N*T,C] 输入序列（量化后，int32_t 存储）
    // h: [(T+1)*N,H] 初始和输出隐藏状态（量化后，int32_t 存储）
    // v: [T*N,H*4] 中间激活值（训练模式需要）
    // zoneout_prob: Zoneout 概率
    // zoneout_mask: [T*N,H] Zoneout mask（int32_t 存储）
    void Run(const int steps, const int32_t *W, const int32_t *R, const int32_t *bx,
             const int32_t *br, const int32_t *x, int32_t *h, int32_t *v,
             const float zoneout_prob, const int32_t *zoneout_mask);

   private:
    // 内部迭代函数
    // cur_Wx_: 当前时间步的 W @ x 结果（指向 tmp_Wx_ 的偏移）
    void IterateInternal(const int32_t *R, const int32_t *bx, const int32_t *br,
                         const int32_t *h, int32_t *h_out, int32_t *v,
                         const int32_t *cur_Wx_, const float zoneout_prob,
                         const int32_t *zoneout_mask);

    // 计算 W @ x GEMM 并 rescale（输出到 tmp_Wx_）
    void ComputeWx(const int32_t *W, const int32_t *x, int steps);

    // 计算 R @ h GEMM 并 rescale（输出到 tmp_Rh_）
    void ComputeRh(const int32_t *R, const int32_t *h);

    // 预分配内存缓冲区
    void EnsureBuffersAllocated(int steps);

    // 预计算权重相关的常量
    void PrecomputeWeightSums(const int32_t *W, const int32_t *R);

    struct private_data;
    private_data *data_;

    QuantGRUReScale rescale_param_;

    // 预分配的内部缓冲区（使用 dev::vector 自动管理内存）
    int max_steps_ = 0;

    // GEMM 结果（int32，rescale 后供 gate 计算使用）
    dev::vector<int32_t> tmp_Wx_;  // [hidden*3 * max_steps * batch]
    dev::vector<int32_t> tmp_Rh_;  // [hidden*3 * batch]

    // 权重和常量（预计算）
    dev::vector<int64_t> W_sum_mul_x_zp_;  // [hidden*3]
    dev::vector<int64_t> R_sum_mul_h_zp_;  // [hidden*3]
    bool weight_sums_computed_ = false;

    // 缓存的权重指针（用于检测权重是否变化）
    const int32_t *cached_W_ = nullptr;
    const int32_t *cached_R_ = nullptr;

    // INT8 GEMM 优化：临时缓冲区（当位宽 <= 8 时使用 cuBLAS INT8 GEMM）
    dev::vector<int8_t> tmp_W_i8_;   // [hidden*3 * input] 权重 int8 缓存
    dev::vector<int8_t> tmp_R_i8_;   // [hidden*3 * hidden] 递归权重 int8 缓存
    dev::vector<int8_t> tmp_x_i8_;   // [input * steps * batch] 输入 int8 缓存
    dev::vector<int8_t> tmp_h_i8_;   // [hidden * batch] 隐藏状态 int8 缓存
    
    // cuBLAS INT8 GEMM 的 N 维度填充
    int N_padded_Rh_ = 0;
    dev::vector<int8_t> h_padded_i8_;  // [hidden * N_padded_Rh]
};

}  // namespace gru
