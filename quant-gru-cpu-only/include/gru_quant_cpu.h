#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"
#include "quantize_ops_helper.h"

namespace cpu {

class ForwardPassQuantCPU {
public:
    ForwardPassQuantCPU(bool training, int batch_size, int input_size, int hidden_size);
    ~ForwardPassQuantCPU();

    void setRescaleParam(const GRUQuantParams &params);

    void Run(int steps, const int32_t *W, const int32_t *R, const int32_t *bw, const int32_t *br,
             const int32_t *x, int32_t *h, int32_t *v, float zoneout_prob, const int32_t *zoneout_mask);

private:
    struct PrivateData;
    std::unique_ptr<PrivateData> data_;

    GateQuantParams gate_params_;
    LinearQuantParamsCPU linear_params_;

    int max_steps_ = 0;
    std::vector<int32_t> tmp_weight_ih_linear_;
    std::vector<int32_t> tmp_weight_hh_linear_;
    std::vector<int64_t> W_sum_mul_x_zp_;
    std::vector<int64_t> R_sum_mul_h_zp_;
    bool weight_sums_computed_ = false;
    const int32_t *cached_W_ = nullptr;
    const int32_t *cached_R_ = nullptr;

    void EnsureBuffersAllocated(int steps);
    void PrecomputeWeightSums(const int32_t *W, const int32_t *R);
    void ComputeLinearX(const int32_t *W, const int32_t *x, const int32_t *bw, int steps);
    void ComputeLinearH(const int32_t *R, const int32_t *h, const int32_t *br);
    void IterateInternal(const int32_t *R, const int32_t *br, const int32_t *h,
                         int32_t *h_out, int32_t *v, const int32_t *cur_linear_x,
                         float zoneout_prob, const int32_t *zoneout_mask);
};

}  // namespace cpu
