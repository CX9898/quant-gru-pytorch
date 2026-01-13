#include <cstdio>
#include <vector>
#include "gru_quant_cpu.h"

int main() {
    const int input_size = 40, hidden_size = 64, batch_size = 1, seq_len = 10;
    const int hidden3 = hidden_size * 3;

    printf("=== Fixed-Point GRU CPU Reference Model ===\n");
    printf("Input=%d, Hidden=%d, Batch=%d, SeqLen=%d\n\n", input_size, hidden_size, batch_size, seq_len);

    // 设置量化参数
    GRUQuantParams params;
    params.hidden_ = hidden_size;
    params.bitwidth_config_.setAllBitWidths(8);
    
    const int8_t shift = 7;
    params.shift_x_ = shift; params.zp_x_ = 0;
    params.shift_h_ = shift; params.zp_h_ = 0;
    params.shift_W_.resize(hidden3, shift);
    params.shift_R_.resize(hidden3, shift);
    params.shift_bw_.resize(hidden3, shift * 2);
    params.shift_br_.resize(hidden3, shift * 2);
    params.shift_weight_ih_linear_ = shift; params.zp_weight_ih_linear_ = 0;
    params.shift_weight_hh_linear_ = shift; params.zp_weight_hh_linear_ = 0;
    params.shift_update_gate_input_ = shift; params.zp_update_gate_input_ = 0;
    params.shift_update_gate_output_ = shift; params.zp_update_gate_output_ = 0;
    params.shift_reset_gate_input_ = shift; params.zp_reset_gate_input_ = 0;
    params.shift_reset_gate_output_ = shift; params.zp_reset_gate_output_ = 0;
    params.shift_new_gate_input_ = shift; params.zp_new_gate_input_ = 0;
    params.shift_new_gate_output_ = shift; params.zp_new_gate_output_ = 0;
    params.shift_mul_reset_hidden_ = shift; params.zp_mul_reset_hidden_ = 0;
    params.shift_mul_new_contribution_ = shift; params.zp_mul_new_contribution_ = 0;
    params.shift_mul_old_contribution_ = shift; params.zp_mul_old_contribution_ = 0;

    // 分配数据
    std::vector<int32_t> W(input_size * hidden3, 0);
    std::vector<int32_t> R(hidden_size * hidden3, 0);
    std::vector<int32_t> bw(hidden3, 0), br(hidden3, 0);
    std::vector<int32_t> x(seq_len * batch_size * input_size, 0);
    std::vector<int32_t> h((seq_len + 1) * batch_size * hidden_size, 0);

    // 运行
    cpu::ForwardPassQuantCPU gru(false, batch_size, input_size, hidden_size);
    gru.setRescaleParam(params);
    gru.Run(seq_len, W.data(), R.data(), bw.data(), br.data(), x.data(), h.data(), nullptr, 0, nullptr);

    printf("Output h[%d]: ", seq_len * batch_size * hidden_size);
    for (int i = 0; i < 5; i++) printf("%d ", h[seq_len * batch_size * hidden_size + i]);
    printf("...\n\n=== Done ===\n");
    return 0;
}
