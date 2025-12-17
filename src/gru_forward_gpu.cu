#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <tuple>
#include <utility>

#include "blas.h"
#include "device_assert.h"
#include "gru.h"
#include "inline_ops.h"
#include "quantize_ops_helper.hpp"
#include "quantized_unit_testing.cuh"

// 调试开关：取消注释以启用调试输出
// #define DEBUG_QUANT

namespace {

namespace op {
template <typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__device__ __forceinline__ void PointwiseOperations(
    int steps_idx, const int batch_dim, const int hidden_dim, const T *Wx, const T *Rh, const T *bx,
    const T *br, const T *h, T *h_out, T *v, const T zoneout_prob, const T *zoneout_mask, T *z_pres,
    T *r_pres,
    T *g_pres) {  // Zoneout mask (only used if ApplyZoneout==true)
    const int row = blockDim.x * blockIdx.x + threadIdx.x;  // 当前线程对应的隐藏单元. hidden_idx
    const int col = blockDim.y * blockIdx.y + threadIdx.y;  // 当前线程对应的batch样本. batch_idx

    if (row >= hidden_dim || col >= batch_dim) return;

    const int weight_idx = col * (hidden_dim * 3) + row;

    // Index into the `h` and `h_out` vectors (they have a stride of `hidden_dim`).
    const int output_idx = col * hidden_dim + row;

    // Indicies into the Wx and Rh matrices (for each of the u, r, and e components).
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

    // Indices into the bias vectors (for each of the u, r, and e components).
    const int bz_idx = row + 0 * hidden_dim;
    const int br_idx = row + 1 * hidden_dim;
    const int bg_idx = row + 2 * hidden_dim;

    const T z_pre = Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx];
    const T z = sigmoid(z_pre);

    const T r_pre = Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx];
    const T r = sigmoid(r_pre);

    const T Rh_add_br_g = Rh[g_idx] + br[bg_idx];
    const T g_pre = Wx[g_idx] + r * Rh_add_br_g + bx[bg_idx];
    const T g = tanh(g_pre);

#ifdef DEBUG_QUANT
    // 调试输出：只在第一个时间步的第一个元素输出
    if (row == 0 && col == 0 && steps_idx == 0) {
        printf("[FLOAT] step=%d: Wx_z=%.6f, Rh_z=%.6f, bx_z=%.6f, br_z=%.6f\n",
               steps_idx, (float)Wx[z_idx], (float)Rh[z_idx], (float)bx[bz_idx], (float)br[bz_idx]);
        printf("[FLOAT]   z_pre=%.6f, z=%.6f\n", (float)z_pre, (float)z);
        printf("[FLOAT]   r_pre=%.6f, r=%.6f\n", (float)r_pre, (float)r);
        printf("[FLOAT]   Rh_add_br_g=%.6f, g_pre=%.6f, g=%.6f\n", 
               (float)Rh_add_br_g, (float)g_pre, (float)g);
        printf("[FLOAT]   h_old=%.6f\n", (float)h[output_idx]);
    }
#endif

    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
    }

    const T old_contrib = z * h[output_idx];
    const T one_minus_z = static_cast<T>(1.0) - z;
    const T new_contrib = one_minus_z * g;
    T cur_h_value = old_contrib + new_contrib;

#ifdef DEBUG_QUANT
    // 调试输出：只在第一个时间步的第一个元素输出
    if (row == 0 && col == 0 && steps_idx == 0) {
        printf("[FLOAT]   old_contrib=%.6f, one_minus_z=%.6f, new_contrib=%.6f, h_new=%.6f\n",
               (float)old_contrib, (float)one_minus_z, (float)new_contrib, (float)cur_h_value);
    }
#endif

    if (ApplyZoneout) {
        if (Training) {
            cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] + h[output_idx];
        } else {
            cur_h_value = (zoneout_prob * h[output_idx]) +
                          ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
        }
    }

    if (Calibration) {
        z_pres[output_idx] = z_pre;
        r_pres[output_idx] = r_pre;
        g_pres[output_idx] = g_pre;
    }

    h_out[output_idx] = cur_h_value;
    //    printf("h_out = %f, z = %f, r = %f, g = %f,z_pre = %f, r_pre = %f, g_pre = %f, h_old =
    //    %f\n", cur_h_value, z, r, g,z_pre,r_pre,g_pre, h[output_idx]); printf("Wx_z = %f, Rh_z =
    //    %f, bx_z = %f, br_z = %f\n", Wx[z_idx], Rh[z_idx], bx[z_idx], br[bz_idx]);
}
}  // namespace op

template <typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(const int batch_dim, const int hidden_dim, const T *Wx,
                                    const T *Rh, const T *bx, const T *br, const T *h, T *h_out,
                                    T *v, const T zoneout_prob, const T *zoneout_mask) {
    op::PointwiseOperations<T, Training, ApplyZoneout, Calibration>(
        0, batch_dim, hidden_dim, Wx, Rh, bx, br, h, h_out, v, zoneout_prob, zoneout_mask, nullptr,
        nullptr, nullptr);
}

template <typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(int steps_idx, const int batch_dim, const int hidden_dim,
                                    const T *Wx, const T *Rh, const T *bx, const T *br, const T *h,
                                    T *h_out, T *v, const T zoneout_prob, const T *zoneout_mask,
                                    T *z_pres, T *r_pres, T *g_pres) {
    op::PointwiseOperations<T, Training, ApplyZoneout, Calibration>(
        steps_idx, batch_dim, hidden_dim, Wx, Rh, bx, br, h, h_out, v, zoneout_prob, zoneout_mask,
        z_pres, r_pres, g_pres);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

template <typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(const int batch_dim, const int hidden_dim, const half *Wx,
                                    const half *Rh, const half *bx, const half *br, const half *h,
                                    half *h_out, half *v, const half zoneout_prob,
                                    const half *zoneout_mask) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}

#endif

}  // anonymous namespace

namespace gru {

template <typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

template <typename T>
ForwardPass<T>::ForwardPass(const bool training, const int batch_size, const int input_size,
                            const int hidden_size, const cublasHandle_t &blas_handle,
                            const cudaStream_t &stream)
    : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template <typename T>
ForwardPass<T>::~ForwardPass() {
    if (data_->sync_stream) {
        cudaEventRecord(data_->event, data_->stream[1]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
        cudaEventRecord(data_->event, data_->stream[0]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    } else {
        cudaStreamSynchronize(data_->stream[1]);
        cudaStreamSynchronize(data_->stream[0]);
    }
    cudaEventDestroy(data_->event);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template <typename T>
void ForwardPass<T>::Iterate(const T *W,   // [C,H*3]
                             const T *R,   // [H,H*3]
                             const T *bx,  // [H*3]
                             const T *br,  // [H*3]
                             const T *x,   // [N,C]
                             const T *h,   // [N,H]
                             T *h_out,     // [N,H]
                             T *v,         // [N,H*4]
                             T *tmp_Wx,    // [N,H*3]
                             T *tmp_Rh,    // [N,H*3]
                             const float zoneout_prob,
                             const T *zoneout_mask) {  // Zoneout mask [N,H]
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, batch_size, input_size,
                  &alpha, W, hidden_size * 3, x, input_size, &beta, tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    IterateInternal(0, R, bx, br, h, h_out, v, tmp_Wx, tmp_Rh, zoneout_prob, zoneout_mask);

    cublasSetStream(blas_handle, save_stream);
}

template <typename T>
void ForwardPass<T>::IterateInternal(int steps_idx,
                                     const T *R,   // [H,H*3]
                                     const T *bx,  // [H*3]
                                     const T *br,  // [H*3]
                                     const T *h,   // [N,H]
                                     T *h_out,     // [N,H]
                                     T *v,         // [N,H*4]
                                     T *tmp_Wx,    // [N,H*3]
                                     T *tmp_Rh,    // [N,H*3]
                                     const float zoneout_prob,
                                     const T *zoneout_mask) {  // Zoneout mask [N,H]
    // Constants for GEMM
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, batch_size, hidden_size,
                  &alpha, R, hidden_size * 3, h, hidden_size, &beta, tmp_Rh, hidden_size * 3);
    
#ifdef DEBUG_QUANT
    // 调试：输出 Rh GEMM 结果前5个值 (第一和第二时间步)
    static int rh_debug_count = 0;
    if (rh_debug_count < 2) {
        cudaDeviceSynchronize();
        T tmp_Rh_host[5];
        T h_host[5];
        cudaMemcpy(tmp_Rh_host, tmp_Rh, sizeof(T) * 5, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_host, h, sizeof(T) * 5, cudaMemcpyDeviceToHost);
        printf("[FLOAT GEMM step=%d] h[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n", steps_idx,
               (float)h_host[0], (float)h_host[1], (float)h_host[2], (float)h_host[3], (float)h_host[4]);
        printf("[FLOAT GEMM step=%d] Rh[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n", steps_idx,
               (float)tmp_Rh_host[0], (float)tmp_Rh_host[1], (float)tmp_Rh_host[2],
               (float)tmp_Rh_host[3], (float)tmp_Rh_host[4]);
        rh_debug_count++;
    }
#endif

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    const int offset = steps_idx * batch_size * hidden_size;

    if (calibration_mode_) {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, true, true, true><<<gridDim, blockDim, 0, stream1>>>(
                steps_idx, batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
                zoneout_prob, zoneout_mask, z_pres_.data() + offset, r_pres_.data() + offset,
                g_pres_.data() + offset);
        } else {
            PointwiseOperations<T, true, false, true><<<gridDim, blockDim, 0, stream1>>>(
                steps_idx, batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
                nullptr, z_pres_.data() + offset, r_pres_.data() + offset, g_pres_.data() + offset);
        }
        return;
    }

    if (training) {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, true, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br,
                                                    h, h_out, v, zoneout_prob, zoneout_mask);
        } else {
            PointwiseOperations<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f, nullptr);
        }
    } else {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, false, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br,
                                                    h, h_out, nullptr, zoneout_prob, zoneout_mask);
        } else {
            PointwiseOperations<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr, 0.0f, nullptr);
        }
    }
}

// 辅助函数：计算向量的 min/max
template <typename T>
inline std::pair<T, T> computeMinMax(const std::vector<T> &data) {
    T min_val = data[0];
    T max_val = data[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
    for (size_t i = 1; i < data.size(); ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    return {min_val, max_val};
}

// 辅助函数：更新范围（取并集）
inline void updateRange(float &min_out, float &max_out, float min_val, float max_val) {
    min_out = std::min(min_out, min_val);
    max_out = std::max(max_out, max_val);
}

// 辅助函数：检查范围是否已初始化
inline bool isRangeUninitialized(float min_val, float max_val) {
    return min_val == std::numeric_limits<float>::max() &&
           max_val == std::numeric_limits<float>::lowest();
}

// 辅助函数：平滑更新范围（EMA）
// 如果未初始化，直接使用当前值；否则使用 90% 旧值 + 10% 新值
inline void updateRangeEMA(float &min_out, float &max_out, float min_val, float max_val,
                           float decay = 0.9f) {
    if (isRangeUninitialized(min_out, max_out)) {
        // 第一次更新，直接赋值
        min_out = min_val;
        max_out = max_val;
    } else {
        // 平滑更新：90% 旧值 + 10% 新值
        min_out = decay * min_out + (1.0f - decay) * min_val;
        max_out = decay * max_out + (1.0f - decay) * max_val;
    }
}

template <typename T>
void updateRangesFromV(const std::vector<T> &h_host, const T *v_dev, size_t steps,
                       size_t hidden_size, size_t batch_size,
                       GRUQuantizationRanges &quant_ranges) {
    std::vector<T> v_host = d2h(v_dev, steps * batch_size * hidden_size * 4);
    const size_t output_size = steps * batch_size * hidden_size;

    std::vector<T> z_out(output_size);
    std::vector<T> r_out(output_size);
    std::vector<T> g_out(output_size);
    std::vector<T> Rh_add_br_g(output_size);
    std::vector<T> rRh_g(output_size);
    std::vector<T> new_contrib(output_size);
    std::vector<T> old_contrib(output_size);

#pragma omp parallel for
    for (int t = 0; t < steps; ++t) {
        const size_t offset_v_per_step = t * batch_size * hidden_size * 4;
        for (int b = 0; b < batch_size; ++b) {
            const size_t offset_v_per_batch = b * hidden_size * 4;
            const size_t offset_v = offset_v_per_step + offset_v_per_batch;
            for (int h = 0; h < hidden_size; ++h) {
                const T z_val = v_host[offset_v + hidden_size * 0 + h];
                const T r_val = v_host[offset_v + hidden_size * 1 + h];
                const T g_val = v_host[offset_v + hidden_size * 2 + h];
                const T Rh_add_br_g_val = v_host[offset_v + hidden_size * 3 + h];
                const T rRh_g_val = r_val * Rh_add_br_g_val;
                const T one_minus_update_val = 1 - z_val;
                const T new_contrib_val = one_minus_update_val * g_val;

                const size_t offset_h = t * batch_size * hidden_size + b * hidden_size + h;
                const T h_old = h_host[offset_h];
                const T old_contrib_val = z_val * h_old;

                z_out[offset_h] = z_val;
                r_out[offset_h] = r_val;
                g_out[offset_h] = g_val;
                Rh_add_br_g[offset_h] = Rh_add_br_g_val;
                rRh_g[offset_h] = rRh_g_val;
                new_contrib[offset_h] = new_contrib_val;
                old_contrib[offset_h] = old_contrib_val;
            }
        }
    }

    // 计算并更新各中间结果的范围
    auto [min_z, max_z] = computeMinMax(z_out);
    updateRange(quant_ranges.min_z_out_, quant_ranges.max_z_out_, min_z, max_z);

    auto [min_r, max_r] = computeMinMax(r_out);
    updateRange(quant_ranges.min_r_out_, quant_ranges.max_r_out_, min_r, max_r);

    auto [min_g, max_g] = computeMinMax(g_out);
    updateRange(quant_ranges.min_g_out_, quant_ranges.max_g_out_, min_g, max_g);

    auto [min_Rh_add_br, max_Rh_add_br] = computeMinMax(Rh_add_br_g);
    updateRange(quant_ranges.min_Rh_add_br_g_, quant_ranges.max_Rh_add_br_g_, min_Rh_add_br, max_Rh_add_br);

    auto [min_rRh, max_rRh] = computeMinMax(rRh_g);
    updateRange(quant_ranges.min_rRh_, quant_ranges.max_rRh_, min_rRh, max_rRh);

    // 注意: one_minus_update 不再单独记录范围，直接复用 z_out 的 scale

    auto [min_new, max_new] = computeMinMax(new_contrib);
    updateRange(quant_ranges.min_new_contrib_, quant_ranges.max_new_contrib_, min_new, max_new);

    auto [min_old, max_old] = computeMinMax(old_contrib);
    updateRange(quant_ranges.min_old_contrib_, quant_ranges.max_old_contrib_, min_old, max_old);
}

// 辅助函数：计算设备端数据的 min/max
template <typename T>
inline std::pair<T, T> computeMinMaxDev(const T *data_dev, size_t size) {
    std::vector<T> data_host = d2h(data_dev, size);
    return computeMinMax(data_host);
}

// 辅助函数：分时间步计算设备端数据的 min/max 并使用 EMA 更新范围
template <typename T>
inline void computeMinMaxPerStepEMA(const T *data_dev, int steps, int step_size,
                                    float &min_out, float &max_out, float decay = 0.9f) {
    // 一次性拷贝所有数据
    std::vector<T> data_host = d2h(data_dev, steps * step_size);

    // 分时间步计算 min/max 并使用 EMA 更新
    for (int t = 0; t < steps; ++t) {
        const T *step_data = data_host.data() + t * step_size;
        T min_val = step_data[0];
        T max_val = step_data[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
        for (int i = 1; i < step_size; ++i) {
            min_val = std::min(min_val, step_data[i]);
            max_val = std::max(max_val, step_data[i]);
        }
        updateRangeEMA(min_out, max_out, static_cast<float>(min_val), static_cast<float>(max_val), decay);
    }
}

// 辅助函数：计算 per-channel 的 min/max
template <typename T>
inline void computeMinMaxPerChannel(const T *data_dev, size_t input_size, size_t channel_size,
                                    std::vector<float> &min_out, std::vector<float> &max_out) {
    std::vector<T> data_host = d2h(data_dev, input_size * channel_size);

#pragma omp parallel for
    for (int c = 0; c < channel_size; ++c) {
        T min_val = data_host[c];
        T max_val = data_host[c];
        for (int i = 1; i < input_size; ++i) {
            const T val = data_host[i * channel_size + c];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        min_out[c] = std::min(min_out[c], static_cast<float>(min_val));
        max_out[c] = std::max(max_out[c], static_cast<float>(max_val));
    }
}

template <typename T>
void updateGRUQuantizationRanges(
    const int steps, const int batch_size, const int hidden_size, const int input_size, const T *W,
    const T *R, const T *bx, const T *br, const T *x, const T *h, const T *v, const T *tmp_Wx,
    const T *tmp_Rh, const dev::vector<T> &z_pres_, const dev::vector<T> &r_pres_,
    const dev::vector<T> &g_pres_, GRUQuantizationRanges &quant_ranges_) {
    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;

    // 输入 x 的范围（一次拷贝，分时间步平滑更新）
    computeMinMaxPerStepEMA(x, steps, NI, quant_ranges_.min_x_, quant_ranges_.max_x_);

    // 隐藏状态 h 的范围（跳过初始状态 h0，一次拷贝，分时间步平滑更新）
    computeMinMaxPerStepEMA(h + NH, steps, NH, quant_ranges_.min_h_, quant_ranges_.max_h_);

    // 权重 W 的范围（per-channel）
    computeMinMaxPerChannel(W, input_size, hidden_size * 3, quant_ranges_.min_W_, quant_ranges_.max_W_);

    // 权重 R 的范围（per-channel）
    computeMinMaxPerChannel(R, hidden_size, hidden_size * 3, quant_ranges_.min_R_, quant_ranges_.max_R_);

    // Wx 结果的范围
    auto [min_Wx, max_Wx] = computeMinMaxDev(tmp_Wx, steps * batch_size * hidden_size * 3);
    updateRange(quant_ranges_.min_Wx_, quant_ranges_.max_Wx_, min_Wx, max_Wx);

    // Rh 结果的范围
    auto [min_Rh, max_Rh] = computeMinMaxDev(tmp_Rh, steps * batch_size * hidden_size * 3);
    updateRange(quant_ranges_.min_Rh_, quant_ranges_.max_Rh_, min_Rh, max_Rh);

    // 偏置 bx 的范围（per-channel）
    computeMinMaxPerChannel(bx, 1, hidden_size * 3, quant_ranges_.min_bx_, quant_ranges_.max_bx_);

    // 偏置 br 的范围（per-channel）
    computeMinMaxPerChannel(br, 1, hidden_size * 3, quant_ranges_.min_br_, quant_ranges_.max_br_);

    // z 门输入的范围
    auto [min_z_pre, max_z_pre] = computeMinMaxDev(z_pres_.data(), z_pres_.size());
    updateRange(quant_ranges_.min_z_pre_, quant_ranges_.max_z_pre_, min_z_pre, max_z_pre);

    // r 门输入的范围
    auto [min_r_pre, max_r_pre] = computeMinMaxDev(r_pres_.data(), r_pres_.size());
    updateRange(quant_ranges_.min_r_pre_, quant_ranges_.max_r_pre_, min_r_pre, max_r_pre);

    // g 门输入的范围
    auto [min_g_pre, max_g_pre] = computeMinMaxDev(g_pres_.data(), g_pres_.size());
    updateRange(quant_ranges_.min_g_pre_, quant_ranges_.max_g_pre_, min_g_pre, max_g_pre);

    // 从 v 中计算其他中间结果的范围
    std::vector<T> h_host = d2h(h, NH * (steps + 1));
    updateRangesFromV<T>(h_host, v, steps, hidden_size, batch_size, quant_ranges_);
}

template <typename T>
void ForwardPass<T>::Run(const int steps,
                         const T *W,   // [C,H*3]
                         const T *R,   // [H,H*3]
                         const T *bx,  // [H*3]
                         const T *br,  // [H*3]
                         const T *x,   // [N,C]
                         T *h,         // [N,H]
                         T *v,         // [N,H*4]
                         T *tmp_Wx,    // [N,H*3]
                         T *tmp_Rh,    // [N,H*3]
                         const float zoneout_prob,
                         const T *zoneout_mask) {  // Zoneout mask [N,H]
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;    // N
    const int input_size = data_->input_size;    // C
    const int hidden_size = data_->hidden_size;  // H
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    if (calibration_mode_) {
        const size_t size = steps * batch_size * hidden_size;
        z_pres_.resize(size);
        r_pres_.resize(size);
        g_pres_.resize(size);
    }

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                  input_size, &alpha, W, hidden_size * 3, x, input_size, &beta, tmp_Wx,
                  hidden_size * 3);
    
#ifdef DEBUG_QUANT
    // 调试：输出 Wx GEMM 结果前5个值
    static bool first_wx_debug = true;
    if (first_wx_debug) {
        cudaDeviceSynchronize();
        T tmp_Wx_host[5];
        cudaMemcpy(tmp_Wx_host, tmp_Wx, sizeof(T) * 5, cudaMemcpyDeviceToHost);
        printf("[FLOAT GEMM] Wx[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
               (float)tmp_Wx_host[0], (float)tmp_Wx_host[1], (float)tmp_Wx_host[2],
               (float)tmp_Wx_host[3], (float)tmp_Wx_host[4]);
        first_wx_debug = false;
    }
#endif
    
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    for (int i = 0; i < steps; ++i) {
        const int Rh_offset = calibration_mode_ ? i * NH * 3 : 0;
        IterateInternal(i, R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh + Rh_offset, zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
        //        break;
    }

    cublasSetStream(blas_handle, save_stream);

    if (calibration_mode_) {
        // 同步所有 GPU 操作，确保数据计算完成
        cudaDeviceSynchronize();
        quant_ranges_.hidden_ = data_->hidden_size;
        // 更新各算子的 min/max 范围
        updateGRUQuantizationRanges<T>(
            steps, batch_size, hidden_size, input_size, W, R, bx, br, x, h, v, tmp_Wx, tmp_Rh,
            z_pres_, r_pres_, g_pres_, quant_ranges_);
    }
}

// template
// struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

}  // namespace gru
