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

    void setRescaleParam(const GRUQuantParams &parms);

    // 简化的 GRU 前向接口（内部管理临时缓冲区）
    // 所有输入输出数据使用 int32_t 存储，实际值限制在配置的位宽范围内
    //
    // W: [C,H*3] 输入权重矩阵（量化后，int32_t 存储）
    // R: [H,H*3] 循环权重矩阵（量化后，int32_t 存储）
    // bw: [H*3] 输入偏置（量化后）
    // br: [H*3] 循环偏置（量化后）
    // x: [N*T,C] 输入序列（量化后，int32_t 存储）
    // h: [(T+1)*N,H] 初始和输出隐藏状态（量化后，int32_t 存储）
    // v: [T*N,H*4] 中间激活值（训练模式需要）
    // zoneout_prob: Zoneout 概率
    // zoneout_mask: [T*N,H] Zoneout mask（int32_t 存储）
    void Run(const int steps, const int32_t *W, const int32_t *R, const int32_t *bw,
             const int32_t *br, const int32_t *x, int32_t *h, int32_t *v,
             const float zoneout_prob, const int32_t *zoneout_mask);

   private:
    // 内部迭代函数 (Linear 融合版本)
    // cur_linear_x: 当前时间步的 W*x + bw 结果（指向 tmp_linear_x_ 的偏移）
    void IterateInternal(const int32_t *R, const int32_t *br,
                         const int32_t *h, int32_t *h_out, int32_t *v,
                         const int32_t *cur_linear_x, const float zoneout_prob,
                         const int32_t *zoneout_mask);

    // 计算输入 Linear 变换: W*x + bw（输出到 tmp_linear_x_）
    void ComputeLinearX(const int32_t *W, const int32_t *x, const int32_t *bw, int steps);

    // 计算隐状态 Linear 变换: R*h + br（输出到 tmp_linear_h_）
    void ComputeLinearH(const int32_t *R, const int32_t *h, const int32_t *br);

    // 预分配内存缓冲区
    void EnsureBuffersAllocated(int steps);

    // 预计算权重相关的常量
    void PrecomputeWeightSums(const int32_t *W, const int32_t *R);

    struct private_data;
    private_data *data_;

    // -------------------- 量化参数（拆分设计）--------------------
    GateQuantParams gate_params_;        ///< 门计算参数（纯标量，传给 PointwiseOperationsQuant）
    LinearQuantParamsGPU linear_params_; ///< Linear 层参数（per-channel，用于 GEMM）

    // 预分配的内部缓冲区（使用 dev::vector 自动管理内存）
    int max_steps_ = 0;

    // Linear 变换结果（int32，供 gate 计算使用）
    dev::vector<int32_t> tmp_weight_ih_linear_;  // [hidden*3 * max_steps * batch] W*x + bw
    dev::vector<int32_t> tmp_weight_hh_linear_;  // [hidden*3 * batch] R*h + br

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

// ============================================================================
// 浮点存储版量化 GRU 前向传播类
// ============================================================================

/**
 * @brief 量化 GRU 前向传播类（浮点存储版本）
 *
 * 与 ForwardPassQuant 的区别：
 *   - 所有量化值使用 float 存储（值仍是定点整数）
 *   - 使用 cuBLAS SGEMM + 单独的 bias/rescale kernel
 *   - 只使用 real_sigmoid/real_tanh，不用 LUT
 *
 * 适用场景：
 *   - 需要浮点存储格式进行后续处理
 *   - 调试和验证量化精度
 */
class ForwardPassQuantFP {
public:
    /// @brief 构造函数
    /// @param training 是否训练模式（需要保存中间值）
    /// @param batch_size 批大小
    /// @param input_size 输入维度
    /// @param hidden_size 隐藏层维度
    /// @param blas_handle cuBLAS 句柄
    /// @param stream CUDA 流（可选）
    ForwardPassQuantFP(bool training, int batch_size, int input_size,
                       int hidden_size, const cublasHandle_t &blas_handle,
                       const cudaStream_t &stream = 0);

    ~ForwardPassQuantFP();

    /// @brief 设置量化参数（从整数版参数转换）
    void setRescaleParam(const GRUQuantParams &params);

    /// @brief 前向传播
    /// @param steps 时间步数
    /// @param W [C, H*3] 输入权重（float 存储的定点值）
    /// @param R [H, H*3] 循环权重
    /// @param bw [H*3] 输入偏置
    /// @param br [H*3] 循环偏置
    /// @param x [N*T, C] 输入序列
    /// @param h [(T+1)*N, H] 隐状态（输入输出）
    /// @param v [T*N, H*4] 中间值（训练模式）
    /// @param zoneout_prob Zoneout 概率
    /// @param zoneout_mask Zoneout mask
    void Run(int steps,
             const float *W, const float *R,
             const float *bw, const float *br,
             const float *x, float *h, float *v,
             float zoneout_prob, const float *zoneout_mask);

private:
    void ComputeLinearX(const float *W, const float *x, const float *bw, int steps);
    void ComputeLinearH(const float *R, const float *h, const float *br);
    void IterateInternal(const float *R, const float *br,
                         const float *h, float *h_out, float *v,
                         const float *cur_weight_ih_linear,
                         float zoneout_prob, const float *zoneout_mask);
    void EnsureBuffersAllocated(int steps);
    void PrecomputeWeightSums(const float *W, const float *R);

    struct private_data;
    private_data *data_;

    GateQuantParamsFP gate_params_;
    LinearQuantParamsGPUFP linear_params_;

    int max_steps_ = 0;

    // GEMM 原始结果（未 rescale）
    dev::vector<float> tmp_gemm_result_;
    // Linear 变换结果（rescale 后）
    dev::vector<float> tmp_weight_ih_linear_;  // [hidden*3 * max_steps * batch]
    dev::vector<float> tmp_weight_hh_linear_;  // [hidden*3 * batch]

    // 权重和常量（预计算，用于零点补偿）
    dev::vector<double> W_sum_mul_x_zp_;  // [hidden*3]
    dev::vector<double> R_sum_mul_h_zp_;  // [hidden*3]
    bool weight_sums_computed_ = false;

    const float *cached_W_ = nullptr;
    const float *cached_R_ = nullptr;
};

}  // namespace gru
