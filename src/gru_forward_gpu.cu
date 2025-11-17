#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <utility>
#include <tuple>

#include "blas.h"
#include "device_assert.h"
#include "gru.h"
#include "inline_ops.h"

namespace {

namespace op {
template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__device__ __forceinline__ void PointwiseOperations(const int batch_dim,
                                                    const int hidden_dim,
                                                    const T *Wx,
                                                    const T *Rh,
                                                    const T *bx,
                                                    const T *br,
                                                    const T *h,
                                                    T *h_out,
                                                    T *v,
                                                    const T zoneout_prob,
                                                    const T *zoneout_mask,
                                                    T *z_pres,
                                                    T *r_pres,
                                                    T *g_pres,
                                                    T *one_minus_update,
                                                    T *new_contrib,
                                                    T *old_contrib) {  // Zoneout mask (only used if ApplyZoneout==true)
    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元. hidden_idx
    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本. batch_idx

    if (row >= hidden_dim || col >= batch_dim)
        return;

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

    const T g_pre = Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx];
    const T g = tanh(g_pre);

    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
    }

    T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * g;

    if (ApplyZoneout) {
        if (Training) {
            cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] + h[output_idx];
        } else {
            cur_h_value = (zoneout_prob * h[output_idx]) + ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
        }
    }

    if (Calibration) {
        z_pres[output_idx] = z_pre;
        r_pres[output_idx] = r_pre;
        g_pres[output_idx] = g_pre;
        one_minus_update[output_idx] = (static_cast<T>(1.0) - z);
        new_contrib[output_idx] = (static_cast<T>(1.0) - z) * g;
        old_contrib[output_idx] = z * h[output_idx];
    }

    h_out[output_idx] = cur_h_value;
//    printf("h_out = %f, z = %f, r = %f, g = %f,z_pre = %f, r_pre = %f, g_pre = %f, h_old = %f\n", cur_h_value, z, r, g,z_pre,r_pre,g_pre, h[output_idx]);
//    printf("Wx_z = %f, Rh_z = %f, bx_z = %f, br_z = %f\n", Wx[z_idx], Rh[z_idx], bx[z_idx], br[bz_idx]);
}
} // op namespace

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(const int batch_dim,
                                    const int hidden_dim,
                                    const T *Wx,
                                    const T *Rh,
                                    const T *bx,
                                    const T *br,
                                    const T *h,
                                    T *h_out,
                                    T *v,
                                    const T zoneout_prob,
                                    const T *zoneout_mask
) {
    op::PointwiseOperations<T, Training, ApplyZoneout, Calibration>(batch_dim,
                                                                    hidden_dim,
                                                                    Wx,
                                                                    Rh,
                                                                    bx,
                                                                    br,
                                                                    h,
                                                                    h_out,
                                                                    v,
                                                                    zoneout_prob,
                                                                    zoneout_mask,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr);
}

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(const int batch_dim,
                                    const int hidden_dim,
                                    const T *Wx,
                                    const T *Rh,
                                    const T *bx,
                                    const T *br,
                                    const T *h,
                                    T *h_out,
                                    T *v,
                                    const T zoneout_prob,
                                    const T *zoneout_mask,
                                    T *z_pres,
                                    T *r_pres,
                                    T *g_pres,
                                    T *one_minus_update,
                                    T *new_contrib,
                                    T *old_contrib
) {
    op::PointwiseOperations<T, Training, ApplyZoneout, Calibration>(batch_dim,
                                                                    hidden_dim,
                                                                    Wx,
                                                                    Rh,
                                                                    bx,
                                                                    br,
                                                                    h,
                                                                    h_out,
                                                                    v,
                                                                    zoneout_prob,
                                                                    zoneout_mask,
                                                                    z_pres,
                                                                    r_pres,
                                                                    g_pres,
                                                                    one_minus_update,
                                                                    new_contrib,
                                                                    old_contrib);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const half *Wx,
                         const half *Rh,
                         const half *bx,
                         const half *br,
                         const half *h,
                         half *h_out,
                         half *v,
                         const half zoneout_prob,
                         const half *zoneout_mask) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}

#endif

}  // anonymous namespace

namespace gru {

template<typename T>
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

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t &blas_handle,
    const cudaStream_t &stream) : data_(new private_data) {
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

template<typename T>
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

template<typename T>
void ForwardPass<T>::Iterate(
    const T *W,  // [C,H*3]
    const T *R,  // [H,H*3]
    const T *bx, // [H*3]
    const T *br, // [H*3]
    const T *x,  // [N,C]
    const T *h,  // [N,H]
    T *h_out,    // [N,H]
    T *v,        // [N,H*4]
    T *tmp_Wx,   // [N,H*3]
    T *tmp_Rh,   // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
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
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, batch_size, input_size,
                  &alpha,
                  W, hidden_size * 3,
                  x, input_size,
                  &beta,
                  tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    IterateInternal(
        0,
        R,
        bx,
        br,
        h,
        h_out,
        v,
        tmp_Wx,
        tmp_Rh,
        zoneout_prob,
        zoneout_mask);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void ForwardPass<T>::IterateInternal(
    int steps_idx,
    const T *R,  // [H,H*3]
    const T *bx, // [H*3]
    const T *br, // [H*3]
    const T *h,  // [N,H]
    T *h_out,    // [N,H]
    T *v,        // [N,H*4]
    T *tmp_Wx,   // [N,H*3]
    T *tmp_Rh,   // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
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
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, batch_size, hidden_size,
                  &alpha,
                  R, hidden_size * 3,
                  h, hidden_size,
                  &beta,
                  tmp_Rh, hidden_size * 3);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    const int offset = steps_idx * batch_size * hidden_size;

    if (calibration_mode_) {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, true, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                v,
                zoneout_prob,
                zoneout_mask,
                z_pres_.data() + offset,
                r_pres_.data() + offset,
                g_pres_.data() + offset,
                one_minus_update_.data() + offset,
                new_contrib_.data() + offset,
                old_contrib_.data() + offset);
        } else {
            PointwiseOperations<T, true, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                v,
                0.0f,
                nullptr,
                z_pres_.data() + offset,
                r_pres_.data() + offset,
                g_pres_.data() + offset,
                one_minus_update_.data() + offset,
                new_contrib_.data() + offset,
                old_contrib_.data() + offset);
        }
        return;
    }

    if (training) {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                v,
                zoneout_prob,
                zoneout_mask);
        } else {
            PointwiseOperations<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                v,
                0.0f,
                nullptr);
        }
    } else {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                nullptr,
                zoneout_prob,
                zoneout_mask);
        } else {
            PointwiseOperations<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                nullptr,
                0.0f,
                nullptr);
        }
    }
}

//template<typename T, typename QuantT>
//std::pair<float, int32_t> calculateQuantScale(T min_val, T max_val, bool use_symmetric = true) {
//    // 确定量化范围
//    constexpr int32_t quant_min = std::numeric_limits<QuantT>::min();
//    constexpr int32_t quant_max = std::numeric_limits<QuantT>::max();
//
//    // 处理特殊情况：min_val == max_val
//    if (min_val == max_val) {
//        if (std::abs(min_val) < 1e-8f) {
//            // 全零或接近零的情况
//            return std::make_pair(1.0f, 0);
//        } else {
//            // 扩展一个小的范围避免除零
//            // 扩展范围应该是原值的 10%，而不是固定的 0.1
//            T range = std::abs(min_val) * 0.1f;
//            min_val -= range;
//            max_val += range;
//        }
//    }
//
//    // 确保 min_val <= max_val
//    if (min_val > max_val) {
//        std::swap(min_val, max_val);
//    }
//
//    float scale;
//    int32_t zero_point;
//
//    if (use_symmetric) {
//        // 对称量化：zero_point固定为0
//        zero_point = 0;
//
//        // 计算最大绝对值范围
//        T abs_max = std::max(std::abs(min_val), std::abs(max_val));
//
//        if (abs_max == 0) {
//            scale = 1.0f;
//        } else {
//            // 对称量化的scale计算
//            scale = static_cast<float>(abs_max) / static_cast<float>(quant_max);
//        }
//
//        // 确保scale不为零
//        if (scale == 0.0f) {
//            scale = 1.0f;
//        }
//    } else {
//        // 非对称量化
//        // 计算scale
//        // 注意：quant_max - quant_min 可能溢出，使用 int64_t 计算
//        int64_t quant_range = static_cast<int64_t>(quant_max) - static_cast<int64_t>(quant_min);
//        scale = (static_cast<float>(max_val) - static_cast<float>(min_val)) /
//                static_cast<float>(quant_range);
//
//        // 确保scale不为零
//        if (scale == 0.0f) {
//            scale = 1.0f;
//            zero_point = 0;
//        } else {
//            // 计算zero_point
//            zero_point = static_cast<int32_t>(std::round(
//                static_cast<float>(quant_min) - static_cast<float>(min_val) / scale
//            ));
//
//            // 限制zero_point在量化范围内
//            zero_point = std::max(quant_min, std::min(quant_max, zero_point));
//        }
//    }
//
//    return std::make_pair(scale, zero_point);
//}

template<typename T, typename QuantT>
std::pair<float, int32_t> calculateQuantScale(T min_val, T max_val, bool symmetric) {
    // 验证输入范围
    if (min_val > max_val) {
        throw std::invalid_argument("min_val cannot be greater than max_val");
    }

    if (std::isnan(min_val) || std::isnan(max_val) ||
        std::isinf(min_val) || std::isinf(max_val)) {
        throw std::invalid_argument("min_val and max_val must be finite numbers");
    }

    // 处理min_val等于max_val的情况
    if (min_val == max_val) {
        if (min_val == 0) {
            min_val = -1.0;
            max_val = 1.0;
        } else {
            double range = std::abs(min_val) * 0.1;
            min_val -= range;
            max_val += range;
        }
    }

    float scale;
    int32_t zero_point;

    // 获取量化类型的范围
    constexpr int32_t qmin = std::numeric_limits<QuantT>::min();
    constexpr int32_t qmax = std::numeric_limits<QuantT>::max();
    const int32_t quant_range = qmax - qmin;

    if (symmetric) {
        // 对称量化：零点固定为0
        double abs_max = std::max(std::abs(min_val), std::abs(max_val));

        // 对称量化的有效范围是 [-qmax, qmax] 或 [-qmax-1, qmax]，取决于类型
        const int symmetric_range = std::max(-qmin, qmax);
        scale = abs_max / symmetric_range;
        zero_point = 0;

    } else {
        // 非对称量化
        double range = max_val - min_val;
        scale = range / quant_range;

        if (scale == 0) {
            zero_point = 0;
        } else {
            // zero_point = round((0 - min_val) / scale) + qmin
            zero_point = static_cast<int>(std::round(-min_val / scale)) + qmin;

            // 确保零点在量化范围内
            zero_point = std::max(qmin, std::min(qmax, zero_point));
        }
    }

    return std::make_pair(scale, zero_point);
}

// // return: scale, zero_point
// template<typename T>
// std::pair<float, int32_t> calculateQuantScale(T min_val, T max_val, bool use_symmetric = true, bool is_int16 = false) {

//     // 根据目标类型选择量化范围
//     int qmin, qmax;
//     if (is_int16) {
//         if (use_symmetric) {
//             qmin = -32768;
//             qmax = 32767;
//         } else {
//             qmin = 0;
//             qmax = 65535;
//         }
//     } else { // int8
//         if (use_symmetric) {
//             qmin = -128;
//             qmax = 127;
//         } else {
//             qmin = 0;
//             qmax = 255;
//         }
//     }

//     float scale = 1.0f;
//     int32_t zp = 0;
//     if (use_symmetric) {
//         float abs_max = std::max(std::abs(min_val), std::abs(max_val));
//         // 避免除零
//         if (abs_max == 0) abs_max = 1e-6f;
//         scale = abs_max / ((float) qmax);
//         zp = 0;
//     } else {
//         // 非对称量化公式：
//         // quantized = round(real / scale + zero_point)
//         // real = (quantized - zero_point) * scale
//         //
//         // 要满足：min_val 映射到 qmin, max_val 映射到 qmax
//         // qmin = round(min_val / scale + zp) ≈ min_val / scale + zp
//         // qmax = round(max_val / scale + zp) ≈ max_val / scale + zp
//         //
//         // 从这两个等式可以解出：
//         // scale = (max_val - min_val) / (qmax - qmin)
//         // zp = qmin - min_val / scale

//         float denominator = max_val - min_val;
//         if (denominator == 0) denominator = 1e-6f;
//         scale = denominator / (float) (qmax - qmin);

//         // 计算 zero point: zp = qmin - min_val / scale
//         // 注意：这里需要先计算 min_val / scale，然后从 qmin 减去
//         float zp_float = static_cast<float>(qmin) - (min_val / scale);
//         zp = static_cast<int32_t>(std::round(zp_float));

//         // 如果 zp < qmin，说明 min_val > 0，计算出的 zp 为负数
//         // 对于非对称量化，如果数据范围是 [min_val, max_val] 且 min_val > 0，
//         // 我们应该让数据范围映射到 [qmin, qmax]
//         //
//         // 方案：设置 zp = qmin = 0，然后重新计算 scale
//         // 使得 max_val 映射到 qmax，min_val 会映射到某个值 >= qmin
//         // scale = max_val / (qmax - zp) = max_val / qmax
//         if (zp < qmin) {
//             zp = qmin;  // 对于 int8 非对称，qmin = 0
//             // 重新计算 scale，使得 max_val 映射到 qmax
//             // max_val / scale + zp = qmax
//             // scale = max_val / (qmax - zp) = max_val / qmax
//             if (max_val > 1e-6f) {
//                 scale = max_val / static_cast<float>(qmax - zp);
//             }
//             // 验证：min_val 会被映射到 min_val / scale + zp = min_val / scale
//             // 这个值应该 >= qmin = 0（因为 min_val > 0, scale > 0）
//         }

//         // zp截断到[qmin, qmax]
//         if (zp > qmax) zp = qmax;

//         // 调试信息：如果 zp 被调整，输出信息
//         if (zp_float < static_cast<float>(qmin) || zp_float > static_cast<float>(qmax)) {
//             printf("[DEBUG calculateQuantScale] min_val=%f, max_val=%f, scale=%f, zp_float=%f, zp=%d (adjusted)\n",
//                    min_val, max_val, scale, zp_float, zp);
//         }
//     }

//     return std::make_pair(scale, zp);
// }

/**
* 通用(仅host)scale/zp 计算函数
* @param x_dev  -- 设备端输入数据指针
* @param size_per_step -- 每步输入长度
* @param steps -- 步数
* @param use_symmetric -- 是否对称量化
* @param is_int16 -- 是否量化为int16（否则int8）
* @return std::pair<float, int32_t> (scale, zp)
*/
template<typename T, typename QuantT>
std::pair<float, int32_t> calculateXScale(const T *x_dev,
                                          int size_per_step,
                                          int steps,
                                          bool use_symmetric = true) {
    std::vector<T> x_host = d2h(x_dev, steps * size_per_step);
    std::vector<T> min(steps);
    std::vector<T> max(steps);

#pragma omp parallel for
    for (int t = 0; t < steps; ++t) {
        const int offset = t * size_per_step;
        min[t] = x_host[offset];
        max[t] = x_host[offset];
        for (int i = 1; i < size_per_step; ++i) {
            min[t] = std::min(min[t], x_host[offset + i]);
            max[t] = std::max(max[t], x_host[offset + i]);
        }
    }

    T res_min = min[0];
    T res_max = max[0];
    printf("calculateXScale: steps=%d, size_per_step=%d\n", steps, size_per_step);
    printf("  Initial: min[0]=%f, max[0]=%f\n", min[0], max[0]);
    for (int t = 1; t < steps; ++t) {
        res_min = 0.9 * res_min + 0.1 * min[t];
        res_max = 0.9 * res_max + 0.1 * max[t];
        if (t < 3 || t == steps - 1) {
            printf("  t=%d: min[t]=%f, max[t]=%f\n", t, min[t], max[t]);
        }
    }
    printf("  EMA result: res_min = %f, res_max = %f\n", res_min, res_max);

    auto result = calculateQuantScale<T, QuantT>(res_min, res_max, use_symmetric);
    printf("  Calculated: scale=%f, zp=%d\n", result.first, result.second);
    return result;
}

template<typename T, typename QuantT>
std::vector<float> calculateWeightScales(const T *W_dev, int out_dim, int in_dim) {
    // 列主序排列

    std::vector<T> W_host = d2h(W_dev, out_dim * in_dim);

    std::vector<float> scales(out_dim);
    std::vector<T> min(out_dim);
    std::vector<T> max(out_dim);

#pragma omp parallel for
    for (int i = 0; i < out_dim; ++i) {
        min[i] = W_host[i];
        max[i] = W_host[i];
        for (int j = 1; j < in_dim; ++j) {
            min[i] = std::min(min[i], W_host[j * out_dim + i]);
            max[i] = std::max(max[i], W_host[j * out_dim + i]);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < out_dim; ++i) {
        scales[i] = calculateQuantScale<T, QuantT>(min[i], max[i], true).first;
    }
    return scales;
}

template<typename T, typename QuantT>
std::pair<float, int32_t> calculateScale(const T *data_dev,
                                         size_t size,
                                         bool use_symmetric = true) {
    std::vector<T> data_host = d2h(data_dev, size);
    T min_val = data_host[0];
    T max_val = data_host[0];
#pragma omp parallel for reduction(min:min_val, max:max_val)
    for (int i = 1; i < size; ++i) {
        const T val = data_host[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    return calculateQuantScale<T, QuantT>(min_val, max_val, use_symmetric);
}

template<typename T, typename QuantT>
std::vector<float> calculateBiasScale(const T *bx_dev,
                                      size_t size,
                                      bool use_symmetric = true) {
    std::vector<T> bx_host = d2h(bx_dev, size);
    std::vector<float> scales(size);

    // 对于偏置，我们需要计算每个偏置值的 scale
    // 偏置可能是正数或负数，所以使用绝对值
    // 对于对称量化，scale = abs(bias) / quant_max
    // 但这里的问题是：如果传入 (0, bias)，那么 abs_max = abs(bias)
    // 但实际上，偏置的范围应该是 [-abs(bias), abs(bias)]

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        T bias_val = bx_host[i];

        // 检查偏置值是否太小
        if (std::abs(bias_val) < 1e-8f) {
            printf("[WARNING] calculateBiasScale: bias[%d] = %f is too small, using default scale\n",
                   static_cast<int>(i), bias_val);
            scales[i] = 1.0f;  // 使用默认 scale
            continue;
        }

        // 对于偏置的对称量化，范围应该是 [-abs(bias), abs(bias)]
        // 所以传入 min_val = -abs(bias), max_val = abs(bias)
        // 这样 scale = abs(bias) / quant_max
        T abs_bias = std::abs(bias_val);
        auto result = calculateQuantScale<T, QuantT>(-abs_bias, abs_bias, use_symmetric);
        scales[i] = result.first;
    }
    return scales;
}


template<typename T>
void calculateScaleFromV(const T *v_dev,
                         size_t steps,
                         size_t hidden_size,
                         size_t batch_size,
                         GRUQuantitativeParameters &quant_parms) {
    std::vector<T> v_host = d2h(v_dev, steps * batch_size * hidden_size * 4);

    T min_z = std::numeric_limits<T>::max(), max_z = std::numeric_limits<T>::min();
    T min_r = std::numeric_limits<T>::max(), max_r = std::numeric_limits<T>::min();
    T min_g = std::numeric_limits<T>::max(), max_g = std::numeric_limits<T>::min();
    T min_Rh_add_br_g = std::numeric_limits<T>::max(), max_Rh_add_br_g = std::numeric_limits<T>::min();

    for (int t = 0; t < steps; ++t) {
        const size_t offset_per_step = t * batch_size * hidden_size * 4;
        for (int b = 0; b < batch_size; ++b) {
            const size_t offset_per_batch = b * hidden_size * 4;
            const size_t offset = offset_per_step + offset_per_batch;
            for (int h = 0; h < hidden_size; ++h) {
                const T z_val = v_host[offset + hidden_size * 0 + h];
                const T r_val = v_host[offset + hidden_size * 1 + h];
                const T g_val = v_host[offset + hidden_size * 2 + h];
                const T Rh_add_br_g_val = v_host[offset + hidden_size * 3 + h];

                min_z = std::min(min_z, z_val);
                max_z = std::max(max_z, z_val);

                min_r = std::min(min_r, r_val);
                max_r = std::max(max_r, r_val);

                min_g = std::min(min_g, g_val);
                max_g = std::max(max_g, g_val);

                min_Rh_add_br_g = std::min(min_Rh_add_br_g, Rh_add_br_g_val);
                max_Rh_add_br_g = std::max(max_Rh_add_br_g, Rh_add_br_g_val);
            }
        }
    }

    std::tie(quant_parms.scale_z_out_, quant_parms.zp_z_out_) = calculateQuantScale<T, int8_t>(min_z, max_z, false);
    std::tie(quant_parms.scale_r_out_, quant_parms.zp_r_out_) = calculateQuantScale<T, int8_t>(min_r, max_r, false);
    std::tie(quant_parms.scale_g_out_, quant_parms.zp_g_out_) = calculateQuantScale<T, int8_t>(min_g, max_g, false);
    std::tie(quant_parms.scale_Rh_add_br_, quant_parms.zp_Rh_add_br_) =
        calculateQuantScale<T, int8_t>(min_Rh_add_br_g, max_Rh_add_br_g, true);
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T *W,  // [C,H*3]
    const T *R,  // [H,H*3]
    const T *bx, // [H*3]
    const T *br, // [H*3]
    const T *x,  // [N,C]
    T *h,        // [N,H]
    T *v,        // [N,H*4]
    T *tmp_Wx,   // [N,H*3]
    T *tmp_Rh,   // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size; // N
    const int input_size = data_->input_size; // C
    const int hidden_size = data_->hidden_size; // H
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    if (calibration_mode_) {
        const size_t size = steps * batch_size * hidden_size;
        z_pres_.resize(size);
        r_pres_.resize(size);
        g_pres_.resize(size);
        one_minus_update_.resize(size);
        new_contrib_.resize(size);
        old_contrib_.resize(size);
    }

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, steps * batch_size, input_size,
                  &alpha,
                  W, hidden_size * 3,
                  x, input_size,
                  &beta,
                  tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    for (int i = 0; i < steps; ++i) {
        const int Rh_offset = calibration_mode_ ? i * NH * 3 : 0;
        IterateInternal(
            i,
            R,
            bx,
            br,
            h + i * NH,
            h + (i + 1) * NH,
            v + i * NH * 4,
            tmp_Wx + i * NH * 3,
            tmp_Rh + Rh_offset,
            zoneout_prob,
            zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(blas_handle, save_stream);

    if (calibration_mode_) {
        // 同步所有 GPU 操作，确保数据计算完成
        quant_parms_.hidden_ = data_->hidden_size;
        if (!use_int16_quant_) {
            std::tie(quant_parms_.scale_x_, quant_parms_.zp_x_) = calculateXScale<T, int8_t>(x, NH, steps, false);
            printf("quant_parms_.scale_x_ = %f, quant_parms_.zp_x_ = %d\n", quant_parms_.scale_x_, quant_parms_.zp_x_);
            std::tie(quant_parms_.scale_h_, quant_parms_.zp_h_) = calculateXScale<T, int8_t>(h, NH, steps + 1, false);

            quant_parms_.scale_W_ = calculateWeightScales<T, int8_t>(W, hidden_size * 3, input_size);
            quant_parms_.scale_R_ = calculateWeightScales<T, int8_t>(R, hidden_size * 3, hidden_size);

            std::tie(quant_parms_.scale_Wx_, quant_parms_.zp_Wx_) =
                calculateScale<T, int8_t>(tmp_Wx, steps * batch_size * hidden_size * 3, false);
            std::tie(quant_parms_.scale_Rh_, quant_parms_.zp_Rh_) =
                calculateScale<T, int8_t>(tmp_Rh, steps * batch_size * hidden_size * 3, false);

            quant_parms_.scale_bx_ = calculateBiasScale<T, int8_t>(bx, hidden_size * 3, true);
//            std::vector<T> bx_tmp = d2h(bx, hidden_size * 3);
//            for (int i = 0; i < bx_tmp.size(); ++i) {
//                printf("bx[%d] = %f ", i, bx_tmp[i]);
//                if (bx_tmp[i] <= 1e-6) {
//                    printf("Error. 原始bx出错\n");
//                    exit(0);
//                }
//            }
            quant_parms_.scale_br_ = calculateBiasScale<T, int8_t>(br, hidden_size * 3, true);

            std::tie(quant_parms_.scale_z_pre_, quant_parms_.zp_z_pre_) =
                calculateScale<T, int8_t>(z_pres_.data(), z_pres_.size(), false);
            std::tie(quant_parms_.scale_r_pre_, quant_parms_.zp_r_pre_) =
                calculateScale<T, int8_t>(r_pres_.data(), r_pres_.size(), false);
            std::tie(quant_parms_.scale_g_pre_, quant_parms_.zp_g_pre_) =
                calculateScale<T, int8_t>(g_pres_.data(), g_pres_.size(), false);

            calculateScaleFromV(v, steps, hidden_size, batch_size, quant_parms_);

            std::tie(quant_parms_.scale_one_minus_update_, quant_parms_.zp_one_minus_update_) =
                calculateScale<T, int8_t>(one_minus_update_.data(), one_minus_update_.size(), false);

            std::tie(quant_parms_.scale_new_contrib_, quant_parms_.zp_new_contrib_) =
                calculateScale<T, int8_t>(new_contrib_.data(), new_contrib_.size(), false);

            std::tie(quant_parms_.scale_old_contrib_, quant_parms_.zp_old_contrib_) =
                calculateScale<T, int8_t>(old_contrib_.data(), old_contrib_.size(), false);
        }
    }


}

//template
//struct ForwardPass<half>;
template
struct ForwardPass<float>;
template
struct ForwardPass<double>;

}  // namespace gru
