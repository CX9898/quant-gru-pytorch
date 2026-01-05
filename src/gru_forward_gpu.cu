#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <tuple>
#include <utility>

#include "blas.h"
#include "calibration_utils.h"
#include "device_assert.h"
#include "gru.h"
#include "inline_ops.h"
#include "quantize_ops_helper.h"
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
        printf("[FLOAT] step=%d: Wx_z=%.6f, Rh_z=%.6f, bx_z=%.6f, br_z=%.6f\n", steps_idx,
               (float)Wx[z_idx], (float)Rh[z_idx], (float)bx[bz_idx], (float)br[bz_idx]);
        printf("[FLOAT]   z_pre=%.6f, z=%.6f\n", (float)z_pre, (float)z);
        printf("[FLOAT]   r_pre=%.6f, r=%.6f\n", (float)r_pre, (float)r);
        printf("[FLOAT]   Rh_add_br_g=%.6f, g_pre=%.6f, g=%.6f\n", (float)Rh_add_br_g, (float)g_pre,
               (float)g);
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
               (float)h_host[0], (float)h_host[1], (float)h_host[2], (float)h_host[3],
               (float)h_host[4]);
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

// 辅助函数已移至 include/calibration_utils.h

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
        printf("[FLOAT GEMM] Wx[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n", (float)tmp_Wx_host[0],
               (float)tmp_Wx_host[1], (float)tmp_Wx_host[2], (float)tmp_Wx_host[3],
               (float)tmp_Wx_host[4]);
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

    // 校准模式下同步 GPU 操作，确保预激活值数据可用
    // 注意：updateGRUQuantizationRanges 已移至 gru_interface.cc 中独立调用
    if (calibration_mode_) {
        cudaDeviceSynchronize();
    }
}

// template
// struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

}  // namespace gru
