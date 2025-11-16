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
    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本

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

// return: scale, zero_point
template<typename T>
std::pair<float, int32_t> calculateQuantScale(T min_val, T max_val, bool use_symmetric = true, bool is_int16 = false) {

    // 根据目标类型选择量化范围
    int qmin, qmax;
    if (is_int16) {
        if (use_symmetric) {
            qmin = -32768;
            qmax = 32767;
        } else {
            qmin = 0;
            qmax = 65535;
        }
    } else { // int8
        if (use_symmetric) {
            qmin = -128;
            qmax = 127;
        } else {
            qmin = 0;
            qmax = 255;
        }
    }

    float scale = 1.0f;
    int32_t zp = 0;
    if (use_symmetric) {
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        // 避免除零
        if (abs_max == 0) abs_max = 1e-6f;
        scale = abs_max / ((float) qmax);
        zp = 0;
    } else {
        // 保证 (0 - min_val)/scale = qmin, (max_val - min_val)/scale = qmax
        float denominator = max_val - min_val;
        if (denominator == 0) denominator = 1e-6f;
        scale = denominator / (float) (qmax - qmin);
        zp = static_cast<int32_t>(std::round(qmin - min_val / scale));
        // zp截断到[qmin, qmax]
        if (zp < qmin) zp = qmin;
        if (zp > qmax) zp = qmax;
    }

    return std::make_pair(scale, zp);
}

/**
* 通用(仅host)scale/zp 计算函数
* @param x_dev  -- 设备端输入数据指针
* @param size_per_step -- 每步输入长度
* @param steps -- 步数
* @param use_symmetric -- 是否对称量化
* @param is_int16 -- 是否量化为int16（否则int8）
* @return std::pair<float, int32_t> (scale, zp)
*/
template<typename T>
std::pair<float, int32_t> calculateXScale(const T *x_dev,
                                          int size_per_step,
                                          int steps,
                                          bool use_symmetric = true,
                                          bool is_int16 = false) {
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

    float res_min = min[0];
    float res_max = max[0];
    for (int t = 1; t < steps; ++t) {
        res_min = 0.9 * res_min + 0.1 * min[t];
        res_max = 0.9 * res_max + 0.1 * max[t];
    }

    return calculateQuantScale(res_min, res_max, use_symmetric, is_int16);
}

template<typename T>
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
        scales[i] = calculateQuantScale(min[i], max[i], true, false).first;
    }
    return scales;
}

template<typename T>
std::pair<float, int32_t> calculateScale(const T *data_dev,
                                         size_t size,
                                         bool use_symmetric = true,
                                         bool is_int16 = false) {
    std::vector<T> data_host = d2h(data_dev, size);
    T min_val = data_host[0];
    T max_val = data_host[0];
#pragma omp parallel for reduction(min:val_min, max:val_max)
    for (int i = 1; i < size; ++i) {
        min_val = std::min(min_val, data_host[i]);
        max_val = std::max(max_val, data_host[i]);
    }
    return calculateQuantScale(max_val, max_val, use_symmetric, is_int16);
}

template<typename T>
std::vector<float> calculateBiasScale(const T *bx_dev,
                                      size_t size,
                                      bool use_symmetric = true,
                                      bool is_int16 = false) {
    std::vector<T> bx_host = d2h(bx_dev, size);
    std::vector<float> scales(size);
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        scales[i] = calculateQuantScale(static_cast<T>(0), bx_host[i], use_symmetric, is_int16).first;
    }
    return scales;
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
        quant_parms_.hidden_ = data_->hidden_size;
        if (!use_int16_quant_) {
            std::tie(quant_parms_.scale_x_, quant_parms_.zp_x_) = calculateXScale(x, NH, steps, false, false);
            std::tie(quant_parms_.scale_h_, quant_parms_.zp_h_) = calculateXScale(h, NH, steps + 1, false, false);

            quant_parms_.scale_W_ = calculateWeightScales(W, hidden_size * 3, input_size);
            quant_parms_.scale_R_ = calculateWeightScales(R, hidden_size * 3, hidden_size);

            std::tie(quant_parms_.scale_Wx_, quant_parms_.zp_Wx_) =
                calculateScale(tmp_Wx, steps * batch_size * hidden_size * 3, false, false);
            std::tie(quant_parms_.scale_Rh_, quant_parms_.zp_Rh_) =
                calculateScale(tmp_Rh, steps * batch_size * hidden_size * 3, false, false);

            quant_parms_.scale_bx_ = calculateBiasScale(bx, hidden_size * 3, true, false);
            quant_parms_.scale_br_ = calculateBiasScale(br, hidden_size * 3, true, false);

            std::tie(quant_parms_.scale_one_minus_update_, quant_parms_.zp_one_minus_update_) =
                calculateScale(one_minus_update_.data(), one_minus_update_.size());

            std::tie(quant_parms_.scale_new_contrib_, quant_parms_.zp_new_contrib_) =
                calculateScale(new_contrib_.data(), new_contrib_.size());

            std::tie(quant_parms_.scale_old_contrib_, quant_parms_.zp_old_contrib_) =
                calculateScale(old_contrib_.data(), old_contrib_.size());
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
