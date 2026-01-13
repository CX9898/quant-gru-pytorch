#pragma once

#include <vector>
#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"

struct GRUQuantParams {
    OperatorQuantConfig bitwidth_config_;
    int hidden_;
    int8_t shift_x_;
    int32_t zp_x_;
    int8_t shift_h_;
    int32_t zp_h_;

    std::vector<int8_t> shift_W_;
    std::vector<int8_t> shift_R_;
    std::vector<int8_t> shift_bw_;
    std::vector<int8_t> shift_br_;

    int8_t shift_weight_ih_linear_;
    int32_t zp_weight_ih_linear_;
    int8_t shift_weight_hh_linear_;
    int32_t zp_weight_hh_linear_;

    int8_t shift_update_gate_input_;
    int32_t zp_update_gate_input_;
    int8_t shift_reset_gate_input_;
    int32_t zp_reset_gate_input_;
    int8_t shift_new_gate_input_;
    int32_t zp_new_gate_input_;

    int8_t shift_update_gate_output_;
    int32_t zp_update_gate_output_;
    int8_t shift_reset_gate_output_;
    int32_t zp_reset_gate_output_;
    int8_t shift_new_gate_output_;
    int32_t zp_new_gate_output_;

    int8_t shift_mul_reset_hidden_;
    int32_t zp_mul_reset_hidden_;
    int8_t shift_mul_new_contribution_;
    int32_t zp_mul_new_contribution_;
    int8_t shift_mul_old_contribution_;
    int32_t zp_mul_old_contribution_;

    SigmoidLUT sigmoid_update_gate_lut_;
    SigmoidLUT sigmoid_reset_gate_lut_;
    SigmoidLUT tanh_new_gate_lut_;
};

struct GateQuantParams {
    int32_t zp_weight_ih_linear_;
    int32_t zp_weight_hh_linear_;
    int32_t zp_h_;

    int32_t zp_update_gate_input_;
    int32_t zp_update_gate_output_;
    int8_t shift_weight_ih_linear_to_update_gate_input_;
    int8_t shift_weight_hh_linear_to_update_gate_input_;

    int32_t zp_reset_gate_input_;
    int32_t zp_reset_gate_output_;
    int8_t shift_weight_ih_linear_to_reset_gate_input_;
    int8_t shift_weight_hh_linear_to_reset_gate_input_;

    int32_t zp_new_gate_input_;
    int32_t zp_new_gate_output_;
    int8_t shift_reset_gate_mul_hh_to_mul_reset_hidden_;
    int32_t zp_mul_reset_hidden_;
    int8_t shift_weight_ih_linear_to_new_gate_input_;
    int8_t shift_mul_reset_hidden_to_new_gate_input_;

    int32_t quant_one_in_update_gate_scale_;
    int32_t zp_mul_new_contribution_;
    int8_t shift_update_new_to_mul_new_contribution_;
    int32_t zp_mul_old_contribution_;
    int8_t shift_update_h_to_mul_old_contribution_;
    int8_t shift_mul_new_contribution_to_h_;
    int8_t shift_mul_old_contribution_to_h_;

    OperatorQuantConfig bitwidth_config_;
    SigmoidLUT sigmoid_update_gate_lut_;
    SigmoidLUT sigmoid_reset_gate_lut_;
    SigmoidLUT tanh_new_gate_lut_;
};

struct LinearQuantParamsCPU {
    int32_t zp_x_;
    int32_t zp_h_;
    std::vector<int8_t> shift_gemm_x_to_weight_ih_linear_;
    std::vector<int8_t> shift_bw_to_weight_ih_linear_;
    std::vector<int8_t> shift_gemm_h_to_weight_hh_linear_;
    std::vector<int8_t> shift_br_to_weight_hh_linear_;
};
