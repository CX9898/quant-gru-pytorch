#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <tuple>
#include <utility>

#include "blas.h"
#include "device_assert.h"
#include "gru.h"
#include "inline_ops.h"
#include "quantize_ops_helper.hpp"
#include "quantized_unit_testing.cuh"

namespace {

namespace op {
template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__device__ __forceinline__ void PointwiseOperations(int steps_idx,
                                                    const int batch_dim,
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
                                                    T *g_pres) {// Zoneout mask (only used if ApplyZoneout==true)
    const int row = blockDim.x * blockIdx.x + threadIdx.x;      // 当前线程对应的隐藏单元. hidden_idx
    const int col = blockDim.y * blockIdx.y + threadIdx.y;      // 当前线程对应的batch样本. batch_idx

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

    //    if (weight_idx == 1 && steps_idx <= 2) {
    //        printf("haste compute Z: Wx_fp=%f, Rh_fp=%f, bx_fp=%f, br_fp=%f, z_pre_fp=%f, z=%f\n",
    //               Wx[z_idx],
    //               Rh[z_idx],
    //               bx[bz_idx],
    //               br[bz_idx],
    //               z_pre,
    //               z);
    //    }

    const T r_pre = Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx];
    const T r = sigmoid(r_pre);

    //    if (weight_idx == 0) {
    //        printf("haste compute R: Wx_fp=%f, Rh_fp=%f, bx_fp=%f, br_fp=%f, r_pre_fp=%f, r=%f\n",
    //               Wx[r_idx],
    //               Rh[r_idx],
    //               bx[br_idx],
    //               br[br_idx],
    //               r_pre,
    //               r);
    //    }

    const T g_pre = Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx];
    const T g = tanh(g_pre);

    //    if (weight_idx == 0) {
    //        printf("haste compute G: Wx_fp=%f, Rh_fp=%f, bx_fp=%f, br_fp=%f, g_pre_fp=%f, g=%f\n",
    //               Wx[g_idx],
    //               Rh[g_idx],
    //               bx[bg_idx],
    //               br[bg_idx],
    //               g_pre,
    //               g);
    //    }

    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
    }

    T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * g;

    //    if (weight_idx == 1 && steps_idx <= 2) {
    //        printf("haste compute H: z=%f, h_old=%f, old_contrib=%f, one_minus_update=%f, g=%f, new_contrib=%f, h=%f\n",
    //               z,
    //               h[output_idx],
    //               z * h[output_idx],
    //               (static_cast<T>(1.0) - z),
    //               g,
    //               (static_cast<T>(1.0) - z) * g,
    //               cur_h_value);
    //    }

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
    }

    h_out[output_idx] = cur_h_value;
    //    printf("h_out = %f, z = %f, r = %f, g = %f,z_pre = %f, r_pre = %f, g_pre = %f, h_old = %f\n", cur_h_value, z, r, g,z_pre,r_pre,g_pre, h[output_idx]);
    //    printf("Wx_z = %f, Rh_z = %f, bx_z = %f, br_z = %f\n", Wx[z_idx], Rh[z_idx], bx[z_idx], br[bz_idx]);
}
}// namespace op

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
                                    const T *zoneout_mask) {
    op::PointwiseOperations<T, Training, ApplyZoneout, Calibration>(0, batch_dim,
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
                                                                    nullptr);
}

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(int steps_idx,
                                    const int batch_dim,
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
                                    T *g_pres) {
    op::PointwiseOperations<T, Training, ApplyZoneout, Calibration>(steps_idx,
                                                                    batch_dim,
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
                                                                    g_pres);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__ void PointwiseOperations(const int batch_dim,
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

}// anonymous namespace

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
    const T *W, // [C,H*3]
    const T *R, // [H,H*3]
    const T *bx,// [H*3]
    const T *br,// [H*3]
    const T *x, // [N,C]
    const T *h, // [N,H]
    T *h_out,   // [N,H]
    T *v,       // [N,H*4]
    T *tmp_Wx,  // [N,H*3]
    T *tmp_Rh,  // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) {// Zoneout mask [N,H]
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
    const T *R, // [H,H*3]
    const T *bx,// [H*3]
    const T *br,// [H*3]
    const T *h, // [N,H]
    T *h_out,   // [N,H]
    T *v,       // [N,H*4]
    T *tmp_Wx,  // [N,H*3]
    T *tmp_Rh,  // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) {// Zoneout mask [N,H]
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
                steps_idx,
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
                g_pres_.data() + offset);
        } else {
            PointwiseOperations<T, true, false, true><<<gridDim, blockDim, 0, stream1>>>(
                steps_idx,
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
                g_pres_.data() + offset);
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
// 支持int8, int16量化
std::pair<float, int32_t> calculateQuantScale(T min_val, T max_val, bool symmetric, const std::string &name = "") {
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

    if (max_val - min_val <= 1e-6f) {
        printf("Warning! %s: max_val - min_val = %f <= 1e-6f\n", name.c_str(), max_val - min_val);
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
            zero_point = std::round(qmin - min_val / scale);
            zero_point = std::max(qmin, std::min(qmax, zero_point));
        }
    }

    if (scale <= 1e-6f) {
        printf("Warning! %s : scale = %.15f <= 1e-6f, min_val = %.15f, max_val = %.15f\n",
               name.c_str(),
               scale,
               min_val,
               max_val);
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
* @param name -- 调试信息
*/
template<typename T, typename QuantT>
void calculateScalePerSteps(const T *x_dev,
                            const int size_per_step,
                            const int steps,
                            const bool use_symmetric,
                            int32_t &exp2_inv,
                            int32_t &zp,
                            const std::string &name = "") {
    if (size_per_step == 0 || steps == 0) {
        printf("Warning! %s input size = 0\n", name.c_str());
        return;
    }
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
    for (int t = 1; t < steps; ++t) {
        //        // TODO: 修改为原来的方法
        //        res_min = 0.9 * res_min + 0.1 * min[t];
        //        res_max = 0.9 * res_max + 0.1 * max[t];
        res_min = std::min(res_min, min[t]);
        res_max = std::max(res_max, max[t]);
    }

    calibrateQuantParams<T, QuantT>(res_min, res_max, use_symmetric, res_min, res_max, exp2_inv, zp, name);
}

template<typename T, typename QuantT>
std::vector<int32_t> calculateScalesPerChannels(const T *W_dev, int channel_size, int input_size,
                                                const std::string &name = "") {
    // 列主序排列

    std::vector<T> W_host = d2h(W_dev, channel_size * input_size);

    std::vector<int32_t> exp2_inv_per_channels(channel_size);
    std::vector<T> min(channel_size);
    std::vector<T> max(channel_size);

#pragma omp parallel for
    for (int i = 0; i < channel_size; ++i) {
        min[i] = W_host[i];
        max[i] = W_host[i];
        for (int j = 1; j < input_size; ++j) {
            min[i] = std::min(min[i], W_host[j * channel_size + i]);
            max[i] = std::max(max[i], W_host[j * channel_size + i]);
        }
    }

    std::vector<int32_t> zp_tmp(channel_size);
#pragma omp parallel for
    for (int i = 0; i < channel_size; ++i) {
        if (min[i] == max[i]) {
            const float half = std::abs(min[i]);
            min[i] = -half;
            max[i] = half;
        }
        calibrateQuantParams<T, QuantT>(min[i],
                                        max[i],
                                        true,
                                        min[i],
                                        max[i],
                                        exp2_inv_per_channels[i],
                                        zp_tmp[i],
                                        name);
    }
    return exp2_inv_per_channels;
}


template<typename T, typename QuantT>
void calculateScale(const std::vector<T> &data_host,
                    const bool use_symmetric,
                    int32_t &exp2_inv,
                    int32_t &zp,
                    const std::string &name = "") {
    T min_val = data_host[0];
    T max_val = data_host[0];
#pragma omp parallel for reduction(min : min_val, max : max_val)
    for (int i = 1; i < data_host.size(); ++i) {
        const T val = data_host[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    T min_new = min_val;
    T max_new = max_val;
    calibrateQuantParams<T, QuantT>(min_val, max_val, use_symmetric, min_new, max_new, exp2_inv, zp, name);
}

template<typename T, typename QuantT>
void calculateScale(const T *data_dev,
                    const size_t size,
                    const bool use_symmetric,
                    int32_t &exp2_inv,
                    int32_t &zp,
                    const std::string &name = "") {
    std::vector<T> data_host = d2h(data_dev, size);
    calculateScale<T, QuantT>(data_host, use_symmetric, exp2_inv, zp, name);
}

void printParms(const GRUQuantitativeParameters &quant_parms) {

    printf("GRUQuantitativeParameters (量化参数):\n");
    printf("  hidden_ = %d\n", quant_parms.hidden_);
    printf("  exp2_inv_x_ = %d, zp_x_ = %d\n",
           quant_parms.exp2_inv_x_, quant_parms.zp_x_);
    printf("  exp2_inv_h_ = %d, zp_h_ = %d\n",
           quant_parms.exp2_inv_h_, quant_parms.zp_h_);

    printf("  exp2_inv_W_ (size %zu): ", quant_parms.exp2_inv_W_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_W_.size() && i < 5; ++i) {
        printf("%d ", quant_parms.exp2_inv_W_[i]);
    }
    if (quant_parms.exp2_inv_W_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_R_ (size %zu): ", quant_parms.exp2_inv_R_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_R_.size() && i < 5; ++i) {
        printf("%d ", quant_parms.exp2_inv_R_[i]);
    }
    if (quant_parms.exp2_inv_R_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_bx_ (size %zu): ", quant_parms.exp2_inv_bx_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_bx_.size() && i < 5; ++i) {
        printf("%d ", quant_parms.exp2_inv_bx_[i]);
    }
    if (quant_parms.exp2_inv_bx_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_br_ (size %zu): ", quant_parms.exp2_inv_br_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_br_.size() && i < 5; ++i) {
        printf("%d ", quant_parms.exp2_inv_br_[i]);
    }
    if (quant_parms.exp2_inv_br_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_Wx_ = %d, zp_Wx_ = %d \n",
           quant_parms.exp2_inv_Wx_, quant_parms.zp_Wx_);
    printf("  exp2_inv_Rh_ = %d, zp_Rh_ = %d \n",
           quant_parms.exp2_inv_Rh_, quant_parms.zp_Rh_);
    printf("  exp2_inv_z_pre_ = %d, zp_z_pre_ = %d \n",
           quant_parms.exp2_inv_z_pre_, quant_parms.zp_z_pre_);
    printf("  exp2_inv_r_pre_ = %d, zp_r_pre_ = %d\n",
           quant_parms.exp2_inv_r_pre_, quant_parms.zp_r_pre_);
    printf("  exp2_inv_g_pre_ = %d, zp_g_pre_ = %d\n",
           quant_parms.exp2_inv_g_pre_, quant_parms.zp_g_pre_);
    printf("  exp2_inv_z_out_ = %d, zp_z_out_ = %d\n",
           quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_);
    printf("  exp2_inv_r_out_ = %d, zp_r_out_ = %d\n",
           quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_);
    printf("  exp2_inv_g_out_ = %d, zp_g_out_ = %d\n",
           quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_);
    printf("  exp2_inv_Rh_add_br_ = %d, zp_Rh_add_br_ = %d\n",
           quant_parms.exp2_inv_Rh_add_br_, quant_parms.zp_Rh_add_br_);
    printf("  exp2_inv_rRh_ = %d, zp_rRh_ = %d\n",
           quant_parms.exp2_inv_rRh_, quant_parms.zp_rRh_);
    printf("  exp2_inv_one_minus_update_ = %d, zp_one_minus_update_ = %d\n",
           quant_parms.exp2_inv_one_minus_update_,
           quant_parms.zp_one_minus_update_);
    printf("  exp2_inv_new_contrib_ = %d, zp_new_contrib_ = %d\n",
           quant_parms.exp2_inv_new_contrib_,
           quant_parms.zp_new_contrib_);
    printf("  exp2_inv_old_contrib_ = %d, zp_old_contrib_ = %d\n",
           quant_parms.exp2_inv_old_contrib_,
           quant_parms.zp_old_contrib_);
}
template<typename T, typename QuantT>
void calculateScaleFromV(const std::vector<T> &h_host,
                         const T *v_dev,
                         size_t steps,
                         size_t hidden_size,
                         size_t batch_size,
                         GRUQuantitativeParameters &quant_parms) {
    std::vector<T> v_host = d2h(v_dev, steps * batch_size * hidden_size * 4);
    const size_t output_size = steps * batch_size * hidden_size;

    std::vector<T> z_out(output_size);
    std::vector<T> r_out(output_size);
    std::vector<T> g_out(output_size);
    std::vector<T> Rh_add_br_g(output_size);
    std::vector<T> rRh_g(output_size);
    std::vector<T> one_minus_update(output_size);
    std::vector<T> new_contrib(output_size);
    std::vector<T> old_contrib(output_size);

    //#pragma omp parallel for
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
                one_minus_update[offset_h] = one_minus_update_val;
                new_contrib[offset_h] = new_contrib_val;
                old_contrib[offset_h] = old_contrib_val;
            }
        }
    }

    calculateScale<T, QuantT>(z_out, false, quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_, "scale_z_out");

    calculateScale<T, QuantT>(r_out, false, quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_, "scale_r_out");
    calculateScale<T, QuantT>(g_out, true, quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_, "scale_g_out");
    calculateScale<T, QuantT>(Rh_add_br_g,
                              false,
                              quant_parms.exp2_inv_Rh_add_br_,
                              quant_parms.zp_Rh_add_br_,
                              "scale_Rh_add_br_g");
    calculateScale<T, QuantT>(rRh_g, false, quant_parms.exp2_inv_rRh_, quant_parms.zp_rRh_, "scale_rRh_g");
    calculateScale<T, QuantT>(one_minus_update,
                              false,
                              quant_parms.exp2_inv_one_minus_update_,
                              quant_parms.zp_one_minus_update_,
                              "scale_one_minus_update");
    calculateScale<T, QuantT>(new_contrib,
                              false,
                              quant_parms.exp2_inv_new_contrib_,
                              quant_parms.zp_new_contrib_,
                              "scale_new_contrib");
    calculateScale<T, QuantT>(old_contrib,
                              false,
                              quant_parms.exp2_inv_old_contrib_,
                              quant_parms.zp_old_contrib_,
                              "scale_old_contrib");

#ifdef DEBUG
    checkScale<T, QuantT>(z_out, quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_, "scale_z_out");
    checkScale<T, QuantT>(r_out, quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_, "scale_r_out");
    checkScale<T, QuantT>(g_out, quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_, "scale_g_out");
    checkScale<T, QuantT>(Rh_add_br_g, quant_parms.exp2_inv_Rh_add_br_, quant_parms.zp_Rh_add_br_, "scale_Rh_add_br_g");
    checkScale<T, QuantT>(rRh_g, quant_parms.exp2_inv_rRh_, quant_parms.zp_rRh_, "scale_rRh_g");
    checkScale<T, QuantT>(one_minus_update,
                          quant_parms.exp2_inv_one_minus_update_,
                          quant_parms.zp_one_minus_update_,
                          "scale_one_minus_update");
    checkScale<T, QuantT>(new_contrib,
                          quant_parms.exp2_inv_new_contrib_,
                          quant_parms.zp_new_contrib_,
                          "scale_new_contrib");
    checkScale<T, QuantT>(old_contrib,
                          quant_parms.exp2_inv_old_contrib_,
                          quant_parms.zp_old_contrib_,
                          "scale_old_contrib");
#endif
}

template<typename T, typename QuantT>
void calculateGRUQuantitativeParameters(const int steps,
                                        const int batch_size,
                                        const int hidden_size,
                                        const int input_size,
                                        const T *W,
                                        const T *R,
                                        const T *bx,
                                        const T *br,
                                        const T *x,
                                        const T *h,
                                        const T *v,
                                        const T *tmp_Wx,
                                        const T *tmp_Rh,
                                        const dev::vector<T> &z_pres_,
                                        const dev::vector<T> &r_pres_,
                                        const dev::vector<T> &g_pres_,
                                        GRUQuantitativeParameters &quant_parms_) {
    const int NH = batch_size * hidden_size;

    calculateScalePerSteps<T, QuantT>(x,
                                      NH,
                                      steps,
                                      false,
                                      quant_parms_.exp2_inv_x_,
                                      quant_parms_.zp_x_,
                                      "scale_x");

    calculateScalePerSteps<T, QuantT>(h + NH,
                                      NH,
                                      steps,
                                      false,
                                      quant_parms_.exp2_inv_h_,
                                      quant_parms_.zp_h_,
                                      "scale_h");

    quant_parms_.exp2_inv_W_ = calculateScalesPerChannels<T, QuantT>(W,
                                                                     hidden_size * 3,
                                                                     input_size,
                                                                     "scale_W");


    quant_parms_.exp2_inv_R_ = calculateScalesPerChannels<T, QuantT>(R,
                                                                     hidden_size * 3,
                                                                     hidden_size,
                                                                     "scale_R");


    calculateScale<T, QuantT>(tmp_Wx,
                              steps * batch_size * hidden_size * 3,
                              false,
                              quant_parms_.exp2_inv_Wx_,
                              quant_parms_.zp_Wx_,
                              "scale_Wx");


    calculateScale<T, QuantT>(tmp_Rh,
                              steps * batch_size * hidden_size * 3,
                              false,
                              quant_parms_.exp2_inv_Rh_,
                              quant_parms_.zp_Rh_,
                              "scale_Rh");


    quant_parms_.exp2_inv_bx_ = calculateScalesPerChannels<T, QuantT>(bx,
                                                                      hidden_size * 3,
                                                                      1,
                                                                      "scale_bx");

    quant_parms_.exp2_inv_br_ = calculateScalesPerChannels<T, QuantT>(br,
                                                                      hidden_size * 3,
                                                                      1,
                                                                      "scale_br");


    calculateScale<T, QuantT>(z_pres_.data(),
                              z_pres_.size(),
                              false,
                              quant_parms_.exp2_inv_z_pre_,
                              quant_parms_.zp_z_pre_,
                              "scale_z_pre");

    calculateScale<T, QuantT>(r_pres_.data(),
                              r_pres_.size(),
                              false,
                              quant_parms_.exp2_inv_r_pre_,
                              quant_parms_.zp_r_pre_,
                              "scale_r_pre");

    calculateScale<T, QuantT>(g_pres_.data(),
                              g_pres_.size(),
                              false,
                              quant_parms_.exp2_inv_g_pre_,
                              quant_parms_.zp_g_pre_,
                              "scale_g_pre");

    std::vector<T> h_host = d2h(h, NH * (steps + 1));
    calculateScaleFromV<T, QuantT>(h_host, v, steps, hidden_size, batch_size, quant_parms_);

#ifdef DEBUG
    std::vector<T> x_host = d2h(x, NH * steps);
    checkScale<T, QuantT>(x_host,
                          quant_parms_.exp2_inv_x_,
                          quant_parms_.zp_x_,
                          "scale_x");

    checkScale<T, int8_t>(h_host,
                          quant_parms_.exp2_inv_h_,
                          quant_parms_.zp_h_,
                          "scale_h");

    std::vector<T> W_host = d2h(W, hidden_size * 3 * input_size);
    checkScalePerChannel<T, int8_t>(W_host,
                                    hidden_size * 3,
                                    input_size,
                                    quant_parms_.exp2_inv_W_,
                                    "scale_W");
    std::vector<T> R_host = d2h(R, hidden_size * 3 * hidden_size);
    checkScalePerChannel<T, int8_t>(R_host,
                                    hidden_size * 3,
                                    hidden_size,
                                    quant_parms_.exp2_inv_R_,
                                    "scale_R");
    std::vector<T> tmp_Wx_host = d2h(tmp_Wx, steps * batch_size * hidden_size * 3);
    checkScale<T, int8_t>(tmp_Wx_host,
                          quant_parms_.exp2_inv_Wx_,
                          quant_parms_.zp_Wx_,
                          "scale_Wx");
    std::vector<T> tmp_Rh_host = d2h(tmp_Rh, steps * batch_size * hidden_size * 3);
    checkScale<T, QuantT>(tmp_Rh_host,
                          quant_parms_.exp2_inv_Rh_,
                          quant_parms_.zp_Rh_,
                          "scale_Rh");
    std::vector<T> bx_host = d2h(bx, hidden_size * 3);
    checkScalePerChannel<T, QuantT>(bx_host,
                                    hidden_size * 3,
                                    1,
                                    quant_parms_.exp2_inv_bx_,
                                    "scale_bx");
    std::vector<T> br_host = d2h(br, hidden_size * 3);
    checkScalePerChannel<T, QuantT>(br_host,
                                    hidden_size * 3,
                                    1,
                                    quant_parms_.exp2_inv_br_,
                                    "scale_br");

    std::vector<T> z_pres_host = d2h(z_pres_.data(), z_pres_.size());

    checkScale<T, QuantT>(z_pres_host,
                          quant_parms_.exp2_inv_z_pre_,
                          quant_parms_.zp_z_pre_,
                          "scale_z_pre");
    std::vector<T> r_pres_host = d2h(r_pres_.data(), r_pres_.size());
    checkScale<T, QuantT>(r_pres_host,
                          quant_parms_.exp2_inv_r_pre_,
                          quant_parms_.zp_r_pre_,
                          "scale_r_pre");
    std::vector<T> g_pres_host = d2h(g_pres_.data(), g_pres_.size());
    checkScale<T, QuantT>(g_pres_host,
                          quant_parms_.exp2_inv_g_pre_,
                          quant_parms_.zp_g_pre_,
                          "scale_g_pre");

    // print quant_parms
    printParms(quant_parms_);
#endif
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T *W, // [C,H*3]
    const T *R, // [H,H*3]
    const T *bx,// [H*3]
    const T *br,// [H*3]
    const T *x, // [N,C]
    T *h,       // [N,H]
    T *v,       // [N,H*4]
    T *tmp_Wx,  // [N,H*3]
    T *tmp_Rh,  // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) {// Zoneout mask [N,H]
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;  // N
    const int input_size = data_->input_size;  // C
    const int hidden_size = data_->hidden_size;// H
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
        //        break;
    }

    cublasSetStream(blas_handle, save_stream);

    if (calibration_mode_) {
        // 同步所有 GPU 操作，确保数据计算完成
        quant_parms_.hidden_ = data_->hidden_size;
        if (!use_int16_quant_) {
            calculateGRUQuantitativeParameters<T, int8_t>(steps, batch_size, hidden_size, input_size, W, R, bx, br, x, h, v, tmp_Wx, tmp_Rh, z_pres_, r_pres_, g_pres_, quant_parms_);
        } else {
            calculateGRUQuantitativeParameters<T, int16_t>(steps, batch_size, hidden_size, input_size, W, R, bx, br, x, h, v, tmp_Wx, tmp_Rh, z_pres_, r_pres_, g_pres_, quant_parms_);
        }
    }
}

//template
//struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

}// namespace gru
