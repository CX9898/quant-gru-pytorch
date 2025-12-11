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
    //        printf("haste compute H: z=%f, h_old=%f, old_contrib=%f, one_minus_update=%f, g=%f,
    //        new_contrib=%f, h=%f\n",
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

void printParms(const GRUQuantitativeParameters &quant_parms) {
    printf("GRUQuantitativeParameters (量化参数):\n");
    printf("  hidden_ = %d\n", quant_parms.hidden_);
    printf("  exp2_inv_x_ = %d, zp_x_ = %d\n", static_cast<int>(quant_parms.exp2_inv_x_), quant_parms.zp_x_);
    printf("  exp2_inv_h_ = %d, zp_h_ = %d\n", static_cast<int>(quant_parms.exp2_inv_h_), quant_parms.zp_h_);

    printf("  exp2_inv_W_ (size %zu): ", quant_parms.exp2_inv_W_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_W_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_W_[i]));
    }
    if (quant_parms.exp2_inv_W_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_R_ (size %zu): ", quant_parms.exp2_inv_R_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_R_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_R_[i]));
    }
    if (quant_parms.exp2_inv_R_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_bx_ (size %zu): ", quant_parms.exp2_inv_bx_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_bx_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_bx_[i]));
    }
    if (quant_parms.exp2_inv_bx_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_br_ (size %zu): ", quant_parms.exp2_inv_br_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_br_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_br_[i]));
    }
    if (quant_parms.exp2_inv_br_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_Wx_ = %d, zp_Wx_ = %d \n", static_cast<int>(quant_parms.exp2_inv_Wx_), quant_parms.zp_Wx_);
    printf("  exp2_inv_Rh_ = %d, zp_Rh_ = %d \n", static_cast<int>(quant_parms.exp2_inv_Rh_), quant_parms.zp_Rh_);
    printf("  exp2_inv_z_pre_ = %d, zp_z_pre_ = %d \n", static_cast<int>(quant_parms.exp2_inv_z_pre_),
           quant_parms.zp_z_pre_);
    printf("  exp2_inv_r_pre_ = %d, zp_r_pre_ = %d\n", static_cast<int>(quant_parms.exp2_inv_r_pre_),
           quant_parms.zp_r_pre_);
    printf("  exp2_inv_g_pre_ = %d, zp_g_pre_ = %d\n", static_cast<int>(quant_parms.exp2_inv_g_pre_),
           quant_parms.zp_g_pre_);
    printf("  exp2_inv_z_out_ = %d, zp_z_out_ = %d\n", static_cast<int>(quant_parms.exp2_inv_z_out_),
           quant_parms.zp_z_out_);
    printf("  exp2_inv_r_out_ = %d, zp_r_out_ = %d\n", static_cast<int>(quant_parms.exp2_inv_r_out_),
           quant_parms.zp_r_out_);
    printf("  exp2_inv_g_out_ = %d, zp_g_out_ = %d\n", static_cast<int>(quant_parms.exp2_inv_g_out_),
           quant_parms.zp_g_out_);
    printf("  exp2_inv_Rh_add_br_ = %d, zp_Rh_add_br_ = %d\n", static_cast<int>(quant_parms.exp2_inv_Rh_add_br_),
           quant_parms.zp_Rh_add_br_);
    printf("  exp2_inv_rRh_ = %d, zp_rRh_ = %d\n", static_cast<int>(quant_parms.exp2_inv_rRh_), quant_parms.zp_rRh_);
    printf("  exp2_inv_one_minus_update_ = %d, zp_one_minus_update_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_one_minus_update_), quant_parms.zp_one_minus_update_);
    printf("  exp2_inv_new_contrib_ = %d, zp_new_contrib_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_new_contrib_), quant_parms.zp_new_contrib_);
    printf("  exp2_inv_old_contrib_ = %d, zp_old_contrib_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_old_contrib_), quant_parms.zp_old_contrib_);
}
template <typename T>
void calculateScaleFromV(const std::vector<T> &h_host, const T *v_dev, size_t steps,
                         size_t hidden_size, size_t batch_size,
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
                one_minus_update[offset_h] = one_minus_update_val;
                new_contrib[offset_h] = new_contrib_val;
                old_contrib[offset_h] = old_contrib_val;
            }
        }
    }

    // z 门输出的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.z_out_, [&](auto tag) {
        using ZOutT = typename decltype(tag)::type;
        calculateScale<T, ZOutT>(z_out, false, quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_,
                                 "scale_z_out");
    });

    // r 门输出的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.r_out_, [&](auto tag) {
        using ROutT = typename decltype(tag)::type;
        calculateScale<T, ROutT>(r_out, false, quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_,
                                 "scale_r_out");
    });

    // g 门输出的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.g_out_, [&](auto tag) {
        using GOutT = typename decltype(tag)::type;
        calculateScale<T, GOutT>(g_out, true, quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_,
                                 "scale_g_out");
    });

    // Rh + br 的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.Rh_add_br_, [&](auto tag) {
        using RhAddBrT = typename decltype(tag)::type;
        calculateScale<T, RhAddBrT>(Rh_add_br_g, false, quant_parms.exp2_inv_Rh_add_br_,
                                    quant_parms.zp_Rh_add_br_, "scale_Rh_add_br_g");
    });

    // r × Rh 的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.rRh_, [&](auto tag) {
        using rRhT = typename decltype(tag)::type;
        calculateScale<T, rRhT>(rRh_g, false, quant_parms.exp2_inv_rRh_, quant_parms.zp_rRh_,
                                "scale_rRh_g");
    });

    // 1 - z 的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.one_minus_update_, [&](auto tag) {
        using OneMinusUpdateT = typename decltype(tag)::type;
        calculateScale<T, OneMinusUpdateT>(one_minus_update, false, quant_parms.exp2_inv_one_minus_update_,
                                           quant_parms.zp_one_minus_update_, "scale_one_minus_update");
    });

    // (1.0 - z) * g 的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.new_contrib_, [&](auto tag) {
        using NewContribT = typename decltype(tag)::type;
        calculateScale<T, NewContribT>(new_contrib, false, quant_parms.exp2_inv_new_contrib_,
                                       quant_parms.zp_new_contrib_, "scale_new_contrib");
    });

    // z * h[output_idx] 的量化
    dispatchByBitWidth(quant_parms.bitwidth_config_.old_contrib_, [&](auto tag) {
        using OldContribT = typename decltype(tag)::type;
        calculateScale<T, OldContribT>(old_contrib, false, quant_parms.exp2_inv_old_contrib_,
                                       quant_parms.zp_old_contrib_, "scale_old_contrib");
    });

#ifdef DEBUG
    // TODO: DEBUG 模式下的 checkScale 也需要对应改写
    dispatchByBitWidth(quant_parms.bitwidth_config_.z_out_, [&](auto tag) {
        using ZOutT = typename decltype(tag)::type;
        checkScale<T, ZOutT>(z_out, quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_, "scale_z_out");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.r_out_, [&](auto tag) {
        using ROutT = typename decltype(tag)::type;
        checkScale<T, ROutT>(r_out, quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_, "scale_r_out");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.g_out_, [&](auto tag) {
        using GOutT = typename decltype(tag)::type;
        checkScale<T, GOutT>(g_out, quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_, "scale_g_out");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.Rh_add_br_, [&](auto tag) {
        using RhAddBrT = typename decltype(tag)::type;
        checkScale<T, RhAddBrT>(Rh_add_br_g, quant_parms.exp2_inv_Rh_add_br_, quant_parms.zp_Rh_add_br_,
                                "scale_Rh_add_br_g");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.rRh_, [&](auto tag) {
        using rRhT = typename decltype(tag)::type;
        checkScale<T, rRhT>(rRh_g, quant_parms.exp2_inv_rRh_, quant_parms.zp_rRh_, "scale_rRh_g");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.one_minus_update_, [&](auto tag) {
        using OneMinusUpdateT = typename decltype(tag)::type;
        checkScale<T, OneMinusUpdateT>(one_minus_update, quant_parms.exp2_inv_one_minus_update_,
                                       quant_parms.zp_one_minus_update_, "scale_one_minus_update");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.new_contrib_, [&](auto tag) {
        using NewContribT = typename decltype(tag)::type;
        checkScale<T, NewContribT>(new_contrib, quant_parms.exp2_inv_new_contrib_,
                                   quant_parms.zp_new_contrib_, "scale_new_contrib");
    });
    dispatchByBitWidth(quant_parms.bitwidth_config_.old_contrib_, [&](auto tag) {
        using OldContribT = typename decltype(tag)::type;
        checkScale<T, OldContribT>(old_contrib, quant_parms.exp2_inv_old_contrib_,
                                   quant_parms.zp_old_contrib_, "scale_old_contrib");
    });
#endif
}

template <typename T>
void calculateGRUQuantitativeParameters(
    const int steps, const int batch_size, const int hidden_size, const int input_size, const T *W,
    const T *R, const T *bx, const T *br, const T *x, const T *h, const T *v, const T *tmp_Wx,
    const T *tmp_Rh, const dev::vector<T> &z_pres_, const dev::vector<T> &r_pres_,
    const dev::vector<T> &g_pres_, GRUQuantitativeParameters &quant_parms_) {
    const int NH = batch_size * hidden_size;

    // 输入 x 的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.x_, [&](auto tag) {
        using XT = typename decltype(tag)::type;
        calculateScalePerSteps<T, XT>(x, batch_size * input_size, steps, false,
                                      quant_parms_.exp2_inv_x_, quant_parms_.zp_x_, "scale_x");
    });

    // 隐藏状态 h 的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.h_, [&](auto tag) {
        using HT = typename decltype(tag)::type;
        calculateScalePerSteps<T, HT>(h + NH, NH, steps, false, quant_parms_.exp2_inv_h_,
                                      quant_parms_.zp_h_, "scale_h");
    });

    // 权重 W 的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.W_, [&](auto tag) {
        using WT = typename decltype(tag)::type;
        quant_parms_.exp2_inv_W_ =
            calculateScalesPerChannels<T, WT>(W, hidden_size * 3, input_size, "scale_W");
    });

    // 权重 R 的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.R_, [&](auto tag) {
        using RT = typename decltype(tag)::type;
        quant_parms_.exp2_inv_R_ =
            calculateScalesPerChannels<T, RT>(R, hidden_size * 3, hidden_size, "scale_R");
    });

    // Wx 结果的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.Wx_, [&](auto tag) {
        using WxT = typename decltype(tag)::type;
        calculateScale<T, WxT>(tmp_Wx, steps * batch_size * hidden_size * 3, false,
                               quant_parms_.exp2_inv_Wx_, quant_parms_.zp_Wx_, "scale_Wx");
    });

    // Rh 结果的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.Rh_, [&](auto tag) {
        using RhT = typename decltype(tag)::type;
        calculateScale<T, RhT>(tmp_Rh, steps * batch_size * hidden_size * 3, false,
                               quant_parms_.exp2_inv_Rh_, quant_parms_.zp_Rh_, "scale_Rh");
    });

    // 偏置 bx 的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.bx_, [&](auto tag) {
        using BxT = typename decltype(tag)::type;
        quant_parms_.exp2_inv_bx_ =
            calculateScalesPerChannels<T, BxT>(bx, hidden_size * 3, 1, "scale_bx");
    });

    // 偏置 br 的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.br_, [&](auto tag) {
        using BrT = typename decltype(tag)::type;
        quant_parms_.exp2_inv_br_ =
            calculateScalesPerChannels<T, BrT>(br, hidden_size * 3, 1, "scale_br");
    });

    // z 门输入的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.z_pre_, [&](auto tag) {
        using ZPreT = typename decltype(tag)::type;
        calculateScale<T, ZPreT>(z_pres_.data(), z_pres_.size(), false, quant_parms_.exp2_inv_z_pre_,
                                 quant_parms_.zp_z_pre_, "scale_z_pre");
    });

    // r 门输入的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.r_pre_, [&](auto tag) {
        using RPreT = typename decltype(tag)::type;
        calculateScale<T, RPreT>(r_pres_.data(), r_pres_.size(), false, quant_parms_.exp2_inv_r_pre_,
                                 quant_parms_.zp_r_pre_, "scale_r_pre");
    });

    // g 门输入的量化
    dispatchByBitWidth(quant_parms_.bitwidth_config_.g_pre_, [&](auto tag) {
        using GPreT = typename decltype(tag)::type;
        calculateScale<T, GPreT>(g_pres_.data(), g_pres_.size(), false, quant_parms_.exp2_inv_g_pre_,
                                 quant_parms_.zp_g_pre_, "scale_g_pre");
    });

    std::vector<T> h_host = d2h(h, NH * (steps + 1));
    calculateScaleFromV<T>(h_host, v, steps, hidden_size, batch_size, quant_parms_);

#ifdef DEBUG
    std::vector<T> x_host = d2h(x, steps * batch_size * input_size);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.x_, [&](auto tag) {
        using XT = typename decltype(tag)::type;
        checkScale<T, XT>(x_host, quant_parms_.exp2_inv_x_, quant_parms_.zp_x_, "scale_x");
    });

    dispatchByBitWidth(quant_parms_.bitwidth_config_.h_, [&](auto tag) {
        using HT = typename decltype(tag)::type;
        checkScale<T, HT>(h_host, quant_parms_.exp2_inv_h_, quant_parms_.zp_h_, "scale_h");
    });

    std::vector<T> W_host = d2h(W, hidden_size * 3 * input_size);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.W_, [&](auto tag) {
        using WT = typename decltype(tag)::type;
        checkScalePerChannel<T, WT>(W_host, hidden_size * 3, input_size, quant_parms_.exp2_inv_W_,
                                    "scale_W");
    });

    std::vector<T> R_host = d2h(R, hidden_size * 3 * hidden_size);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.R_, [&](auto tag) {
        using RT = typename decltype(tag)::type;
        checkScalePerChannel<T, RT>(R_host, hidden_size * 3, hidden_size, quant_parms_.exp2_inv_R_,
                                    "scale_R");
    });

    std::vector<T> tmp_Wx_host = d2h(tmp_Wx, steps * batch_size * hidden_size * 3);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.Wx_, [&](auto tag) {
        using WxT = typename decltype(tag)::type;
        checkScale<T, WxT>(tmp_Wx_host, quant_parms_.exp2_inv_Wx_, quant_parms_.zp_Wx_, "scale_Wx");
    });

    std::vector<T> tmp_Rh_host = d2h(tmp_Rh, steps * batch_size * hidden_size * 3);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.Rh_, [&](auto tag) {
        using RhT = typename decltype(tag)::type;
        checkScale<T, RhT>(tmp_Rh_host, quant_parms_.exp2_inv_Rh_, quant_parms_.zp_Rh_, "scale_Rh");
    });

    std::vector<T> bx_host = d2h(bx, hidden_size * 3);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.bx_, [&](auto tag) {
        using BxT = typename decltype(tag)::type;
        checkScalePerChannel<T, BxT>(bx_host, hidden_size * 3, 1, quant_parms_.exp2_inv_bx_,
                                     "scale_bx");
    });

    std::vector<T> br_host = d2h(br, hidden_size * 3);
    dispatchByBitWidth(quant_parms_.bitwidth_config_.br_, [&](auto tag) {
        using BrT = typename decltype(tag)::type;
        checkScalePerChannel<T, BrT>(br_host, hidden_size * 3, 1, quant_parms_.exp2_inv_br_,
                                     "scale_br");
    });

    std::vector<T> z_pres_host = d2h(z_pres_.data(), z_pres_.size());
    dispatchByBitWidth(quant_parms_.bitwidth_config_.z_pre_, [&](auto tag) {
        using ZPreT = typename decltype(tag)::type;
        checkScale<T, ZPreT>(z_pres_host, quant_parms_.exp2_inv_z_pre_, quant_parms_.zp_z_pre_,
                             "scale_z_pre");
    });

    std::vector<T> r_pres_host = d2h(r_pres_.data(), r_pres_.size());
    dispatchByBitWidth(quant_parms_.bitwidth_config_.r_pre_, [&](auto tag) {
        using RPreT = typename decltype(tag)::type;
        checkScale<T, RPreT>(r_pres_host, quant_parms_.exp2_inv_r_pre_, quant_parms_.zp_r_pre_,
                             "scale_r_pre");
    });

    std::vector<T> g_pres_host = d2h(g_pres_.data(), g_pres_.size());
    dispatchByBitWidth(quant_parms_.bitwidth_config_.g_pre_, [&](auto tag) {
        using GPreT = typename decltype(tag)::type;
        checkScale<T, GPreT>(g_pres_host, quant_parms_.exp2_inv_g_pre_, quant_parms_.zp_g_pre_,
                             "scale_g_pre");
    });

    // print quant_parms
    printParms(quant_parms_);
#endif
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
        quant_parms_.hidden_ = data_->hidden_size;
        // 每个算子的位宽由 bitwidth_config_ 独立控制
        calculateGRUQuantitativeParameters<T>(
            steps, batch_size, hidden_size, input_size, W, R, bx, br, x, h, v, tmp_Wx, tmp_Rh,
            z_pres_, r_pres_, g_pres_, quant_parms_);
    }
}

// template
// struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

}  // namespace gru
