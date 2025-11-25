#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cstdio>

#include "blas.h"
#include "device_assert.h"
#include "gru_quant.h"
#include "inline_ops.h"

namespace {

template<typename QuantT, bool ApplyZoneout>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const QuantT *h,
                         const QuantT *v,
                         const QuantT *dh_new,
                         int32_t *dbx_out,
                         int32_t *dbr_out,
                         QuantT *dh_inout,
                         QuantT *dp_out,
                         QuantT *dq_out,
                         const QuantT *zoneout_mask) {  // Zoneout mask (only used if ApplyZoneout==true)
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int base_idx = col * hidden_dim + row;

    QuantT dh_total = dh_new[base_idx] + dh_inout[base_idx];

    const int stride4_base_idx = col * (hidden_dim * 4) + row;
    const int z_idx = stride4_base_idx + 0 * hidden_dim;
    const int r_idx = stride4_base_idx + 1 * hidden_dim;
    const int g_idx = stride4_base_idx + 2 * hidden_dim;
    const int q_g_idx = stride4_base_idx + 3 * hidden_dim;

    const QuantT z = v[z_idx];
    const QuantT r = v[r_idx];
    const QuantT g = v[g_idx];
    const QuantT q_g = v[q_g_idx];

//    if (ApplyZoneout) { // TODO: 支持量化
//        const QuantT mask = zoneout_mask[base_idx];
//        dh_inout[base_idx] = (static_cast<QuantT>(1.0) - mask) * dh_total;
//        dh_total = mask * dh_total;
//        dh_inout[base_idx] += z * dh_total;
//    } else {
//        dh_inout[base_idx] = z * dh_total;
//    }

    const QuantT dg = (static_cast<QuantT>(1.0) - z) * dh_total;
    const QuantT dz = (h[base_idx] - g) * dh_total;
    const QuantT dp_g = d_tanh(g) * dg;
    const QuantT dq_g = dp_g * r;
    const QuantT dr = dp_g * q_g;
    const QuantT dp_r = d_sigmoid(r) * dr;
    const QuantT dq_r = dp_r;
    const QuantT dp_z = d_sigmoid(z) * dz;
    const QuantT dq_z = dp_z;

    const int idx = col * (hidden_dim * 3) + row;

    dp_out[idx + 0 * hidden_dim] = dp_z;
    dp_out[idx + 1 * hidden_dim] = dp_r;
    dp_out[idx + 2 * hidden_dim] = dp_g;

    dq_out[idx + 0 * hidden_dim] = dq_z;
    dq_out[idx + 1 * hidden_dim] = dq_r;
    dq_out[idx + 2 * hidden_dim] = dq_g;

    atomicAdd(&dbx_out[row + 0 * hidden_dim], dp_z);
    atomicAdd(&dbx_out[row + 1 * hidden_dim], dp_r);
    atomicAdd(&dbx_out[row + 2 * hidden_dim], dp_g);

    atomicAdd(&dbr_out[row + 0 * hidden_dim], dq_z);
    atomicAdd(&dbr_out[row + 1 * hidden_dim], dq_r);
    atomicAdd(&dbr_out[row + 2 * hidden_dim], dq_g);
}

}  // anonymous namespace

namespace gru {

template<typename QuantT>
struct BackwardPassQuant<QuantT>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template<typename QuantT>
BackwardPassQuant<QuantT>::BackwardPassQuant(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t &blas_handle,
    const cudaStream_t &stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename QuantT>
BackwardPassQuant<QuantT>::~BackwardPassQuant() {
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

template<typename QuantT>
void BackwardPassQuant<QuantT>::Iterate(
    const QuantT *W_t,     // [H*3,C]
    const QuantT *R_t,     // [H*3,H]
    const int32_t *bx,      // [H*3]
    const int32_t *br,      // [H*3]
    const QuantT *x_t,     // [C,N]
    const QuantT *h,       // [N,H]
    const QuantT *v,       // [N,H*4]
    const QuantT *dh_new,  // [N,H]
    QuantT *dx,            // [N,C]
    QuantT *dW,            // [C,H*3]
    QuantT *dR,            // [H,H*3]
    int32_t *dbx,           // [H*3]
    int32_t *dbr,           // [H*3]
    QuantT *dh,            // [N,H]
    QuantT *dp,            // [N,H*3]
    QuantT *dq,            // [N,H*3]
    const QuantT *zoneout_mask) {  // [N,H]
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    static const int32_t alpha = static_cast<int32_t>(1);
    static const int32_t beta_sum = static_cast<int32_t>(1);
    static const int32_t beta_assign = static_cast<int32_t>(0);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int input_size = data_->input_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    IterateInternal(
        R_t,
        h,
        v,
        dh_new,
        dbx,
        dbr,
        dh,
        dp,
        dq,
        zoneout_mask);

    cublasSetStream(blas_handle, stream1);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, input_size, batch_size,
                  &alpha,
                  dp, hidden_size * 3,
                  x_t, batch_size,
                  &beta_sum,
                  dW, hidden_size * 3);

    // Wait for pointwise operations to complete since there's a
    // data dependency between its output (`dp`, `dq`) and the following matmuls.
    cudaStreamWaitEvent(stream2, event, 0);

    cublasSetStream(blas_handle, stream2);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  input_size, batch_size, hidden_size * 3,
                  &alpha,
                  W_t, input_size,
                  dp, hidden_size * 3,
                  &beta_assign,
                  dx, input_size);

    cublasSetStream(blas_handle, stream2);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_T,
                  hidden_size * 3, hidden_size, batch_size,
                  &alpha,
                  dq, hidden_size * 3,
                  h, hidden_size,
                  &beta_sum,
                  dR, hidden_size * 3);

    cublasSetStream(blas_handle, save_stream);
}

template<typename QuantT>
void BackwardPassQuant<QuantT>::IterateInternal( // 内部迭代
    const QuantT *R_t,     // [H*3,H]
    const QuantT *h,       // [N,H]
    const QuantT *v,       // [N,H*4]
    const QuantT *dh_new,  // [N,H]
    int32_t *dbx,           // [H*3]
    int32_t *dbr,           // [H*3]
    QuantT *dh,            // [N,H]
    QuantT *dp,            // [N,H*3]
    QuantT *dq,            // [N,H*3]
    const QuantT *zoneout_mask) {  // [N,H]

    static const int32_t alpha = static_cast<int32_t>(1);
    static const int32_t beta_sum = static_cast<int32_t>(1);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    if (zoneout_mask) {
        PointwiseOperations<QuantT, true><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            h,
            v,
            dh_new,
            dbx,
            dbr,
            dh,
            dp,
            dq,
            zoneout_mask
        );
    } else {
        PointwiseOperations<QuantT, false><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            h,
            v,
            dh_new,
            dbx,
            dbr,
            dh,
            dp,
            dq,
            nullptr
        );
    }
    cudaEventRecord(event, stream1);

    cublasSetStream(blas_handle, stream1);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size * 3,
                  &alpha,
                  R_t, hidden_size,
                  dq, hidden_size * 3,
                  &beta_sum,
                  dh, hidden_size);
}

template<typename QuantT>
void BackwardPassQuant<QuantT>::Run(
    const int steps,
    const QuantT *W_t, // 输入权重转置
    const QuantT *R_t, // 循环权重转置
    const int32_t *bx, // 输入bias
    const int32_t *br, // 循环bias
    const QuantT *x_t, // 当前步输入
    const QuantT *h, // 前一时刻隐藏状态
    const QuantT *v, // 前向传播中间缓存(含z, r,g,h_out)
    const QuantT *dh_new, // 从后面时间步传来的梯度
    QuantT *dx, //
    QuantT *dW,
    QuantT *dR,
    int32_t *dbx,
    int32_t *dbr,
    QuantT *dh,
    QuantT *dp,
    QuantT *dq,
    const QuantT *zoneout_mask) {
    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    using int32_t = std::conditional_t<
        std::is_same_v<QuantT, int8_t> || std::is_same_v<QuantT, int16_t>,
        int,
        QuantT>;

    const int32_t alpha = static_cast<int32_t>(1);
    const int32_t beta_sum = static_cast<int32_t>(1);
    const int32_t beta_assign = static_cast<int32_t>(0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    const int NH = batch_size * hidden_size;
    for (int i = steps - 1; i >= 0; --i) { // 反向迭代
        IterateInternal(
            R_t,
            h + i * NH,
            v + i * NH * 4,
            dh_new + (i + 1) * NH,
            dbx,
            dbr,
            dh,
            dp + i * NH * 3,
            dq + i * NH * 3,
            zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    // Wait for pointwise operations to complete since there's a
    // data dependency between its output (`dp`, `dq`) and the following matmuls.
    cudaStreamWaitEvent(stream2, event, 0);

    cublasSetStream(blas_handle, stream2);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  input_size, batch_size * steps, hidden_size * 3,
                  &alpha,
                  W_t, input_size,
                  dp, hidden_size * 3,
                  &beta_assign,
                  dx, input_size);

    cublasSetStream(blas_handle, stream2);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_T,
                  hidden_size * 3, hidden_size, batch_size * steps,
                  &alpha,
                  dq, hidden_size * 3,
                  h, hidden_size,
                  &beta_sum,
                  dR, hidden_size * 3);

    cublasSetStream(blas_handle, stream1);
    blas<QuantT>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, input_size, batch_size * steps,
                  &alpha,
                  dp, hidden_size * 3,
                  x_t, batch_size * steps,
                  &beta_sum,
                  dW, hidden_size * 3);

    cublasSetStream(blas_handle, save_stream);
}

template
struct BackwardPassQuant<int8_t>;
template
struct BackwardPassQuant<int16_t>;

}  // namespace gru


