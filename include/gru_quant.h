#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "quantize_ops_helper.hpp"

namespace gru {

template <typename QuantT>
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

    // Performs one forward iteration of the GRU cell.
    //
    // W: [C,H*3] the input weight matrix.
    // R: [H,H*3] the recurrent weight matrix.
    // bx: [H*3] the bias for the input weight matrix.
    // br: [H*3] the bias for the recurrent weight matrix.
    // x: [N,C] the GRU input for this iteration (N vectors, each with dimension C).
    // h: [N,H] the t-1 iteration's `h_out` or the initial hidden state if this is the
    //     t=0 iteration (typically zeros).
    // h_out: [N,H] the GRU's output, and the input to the next iteration's `h`. This
    //     pointer may be the same as `h`. Each iteration may reuse the same memory region.
    // v: [N,H*4] if `training` is `false`, this can be a null pointer. If `training` is
    //     `true`, this vector will contain intermediate activations for this iteration which
    //     must be provided as-is to the corresponding backward iteration. The caller must
    //     provide a new memory region for each iteration.
    //     注意: v 使用 int32_t 统一存储，但内部各部分原始类型不同：
    //       - v[0*H : 1*H]: z (QuantZ 类型，存为 int32_t)
    //       - v[1*H : 2*H]: r (QuantR 类型，存为 int32_t)
    //       - v[2*H : 3*H]: g (QuantG 类型，存为 int32_t)
    //       - v[3*H : 4*H]: Rh_add_br_g (int32_t 类型)
    //     反量化时需使用对应的量化参数 (exp2_inv_z/r/g/Rh_add_br, zp_z/r/g/Rh_add_br)。
    // tmp_Wx: [N,H*3] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector, and must provide a new memory region for
    //     each iteration.
    // tmp_Rh: [N,H*3] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector. The same memory region may be provided
    //     for each iteration.
    // zoneout_prob: 0.0 <= zoneout_prob <= 1.0; specifies the probability of a hidden
    //     activation being randomly zoned out. If zoneout was used during training, this
    //     parameter must also be specified during inference with the same value.
    // zoneout_mask: [N,H] may be null to disable zoneout. This is a random binary mask
    //     following a Bernoulli(1-zoneout_prob) distribution. A different mask is typically
    //     used for each iteration.
    void Iterate(const QuantT *W, const QuantT *R, const int32_t *bx, const int32_t *br,
                 const QuantT *x, const QuantT *h, QuantT *h_out, int32_t *v, int32_t *tmp_Wx,
                 int32_t *tmp_Rh, const float zoneout_prob, const QuantT *zoneout_mask);

    void Run(const int steps, const QuantT *W, const QuantT *R, const int32_t *bx,
             const int32_t *br, const QuantT *x, QuantT *h, int32_t *v, int32_t *tmp_Wx,
             int32_t *tmp_Rh, const float zoneout_prob, const QuantT *zoneout_mask);

   private:
    void IterateInternal(const QuantT *R, const int32_t *bx, const int32_t *br, const QuantT *h,
                         QuantT *h_out, int32_t *v, const int32_t *tmp_Wx, int32_t *tmp_Rh,
                         const int *W_sum_mul_x_zp,  // hidden_size * 3
                         const int *R_sum_mul_h_zp,  // hidden_size * 3
                         const float zoneout_prob, const QuantT *zoneout_mask);

    struct private_data;
    private_data *data_;

    QuantGRUReScale rescale_param_;
};

}  // namespace gru
