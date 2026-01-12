// =====================================================================
// 直方图校准辅助函数实现（histogram_calibration_utils.cu）
// =====================================================================

#include "histogram_calibration_utils.h"

#include <cuda_runtime.h>

#include <vector>

#include "dev_vector.h"

// =====================================================================
// CPU 直方图收集实现
// =====================================================================

void collectAllHistograms(
    GRUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *Wx_add_bx, const float *Rh_add_br,
    const float *W, const float *R, const float *bx, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size) {
    
    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;

    // 1. 收集输入 x 的直方图
    collectHistogramPerStep(hist_collectors.x_hist, x, time_steps, NI);

    // 2. 收集隐藏状态 h 的直方图（跳过初始状态）
    collectHistogramPerStep(hist_collectors.h_hist, h + NH, time_steps, NH);

    // 3. 收集 Wx+bx 结果的直方图
    collectHistogramPerStep(hist_collectors.Wx_hist, Wx_add_bx, time_steps, NH * 3);

    // 4. 收集 Rh+br 结果的直方图
    collectHistogramPerStep(hist_collectors.Rh_hist, Rh_add_br, time_steps, NH * 3);

    // 5. 收集权重的 per-channel 直方图（只在首次收集）
    if (!hist_collectors.W_hist[0].is_valid()) {
        collectPerChannelHistograms(hist_collectors.W_hist, W, input_size, hidden_size * 3);
        collectPerChannelHistograms(hist_collectors.R_hist, R, hidden_size, hidden_size * 3);
        collectPerChannelHistograms(hist_collectors.bx_hist, bx, 1, hidden_size * 3);
        collectPerChannelHistograms(hist_collectors.br_hist, br, 1, hidden_size * 3);
    }

    // 6. 从 v 中收集门的中间值直方图
    // v 布局: [T, B, H*4] = [z, r, g, Rh_add_br_g]
    std::vector<float> v_host = d2h(v, time_steps * batch_size * hidden_size * 4);
    std::vector<float> h_host = d2h(h, (time_steps + 1) * batch_size * hidden_size);

    const size_t output_size = time_steps * batch_size * hidden_size;
    std::vector<float> update_gate_out(output_size);
    std::vector<float> reset_gate_out(output_size);
    std::vector<float> new_gate_out(output_size);
    std::vector<float> Rh_add_br_g(output_size);
    std::vector<float> mul_reset_hidden(output_size);
    std::vector<float> mul_new_contribution(output_size);
    std::vector<float> mul_old_contribution(output_size);

    // 解析 v 中的值
    for (int t = 0; t < time_steps; ++t) {
        for (int b = 0; b < batch_size; ++b) {
            const size_t v_base = t * batch_size * hidden_size * 4 + b * hidden_size * 4;
            const size_t out_base = t * batch_size * hidden_size + b * hidden_size;

            for (int hh = 0; hh < hidden_size; ++hh) {
                const float update_val = v_host[v_base + 0 * hidden_size + hh];
                const float reset_val = v_host[v_base + 1 * hidden_size + hh];
                const float new_val = v_host[v_base + 2 * hidden_size + hh];
                const float Rh_add_br_val = v_host[v_base + 3 * hidden_size + hh];

                update_gate_out[out_base + hh] = update_val;
                reset_gate_out[out_base + hh] = reset_val;
                new_gate_out[out_base + hh] = new_val;
                Rh_add_br_g[out_base + hh] = Rh_add_br_val;
                mul_reset_hidden[out_base + hh] = reset_val * Rh_add_br_val;
                mul_new_contribution[out_base + hh] = (1.0f - update_val) * new_val;

                // h_old 是上一个时间步的隐藏状态
                const size_t h_base = t * batch_size * hidden_size + b * hidden_size;
                mul_old_contribution[out_base + hh] = update_val * h_host[h_base + hh];
            }
        }
    }

    // 分时间步收集直方图
    for (int t = 0; t < time_steps; ++t) {
        const float *update_gate_step = update_gate_out.data() + t * batch_size * hidden_size;
        const float *reset_gate_step = reset_gate_out.data() + t * batch_size * hidden_size;
        const float *new_gate_step = new_gate_out.data() + t * batch_size * hidden_size;
        const float *Rh_add_br_step = Rh_add_br_g.data() + t * batch_size * hidden_size;
        const float *mul_reset_hidden_step = mul_reset_hidden.data() + t * batch_size * hidden_size;
        const float *mul_new_contribution_step = mul_new_contribution.data() + t * batch_size * hidden_size;
        const float *mul_old_contribution_step = mul_old_contribution.data() + t * batch_size * hidden_size;

        hist_collectors.update_gate_output_hist.collect(update_gate_step, NH);
        hist_collectors.reset_gate_output_hist.collect(reset_gate_step, NH);
        hist_collectors.new_gate_output_hist.collect(new_gate_step, NH);
        hist_collectors.Rh_add_br_g_hist.collect(Rh_add_br_step, NH);
        hist_collectors.mul_reset_hidden_hist.collect(mul_reset_hidden_step, NH);
        hist_collectors.mul_new_contribution_hist.collect(mul_new_contribution_step, NH);
        hist_collectors.mul_old_contribution_hist.collect(mul_old_contribution_step, NH);
    }

    // 7. 收集 gate input 的直方图（如果提供）
    if (pres_size > 0 && z_pres && r_pres && g_pres) {
        std::vector<float> update_gate_input_host(pres_size);
        std::vector<float> reset_gate_input_host(pres_size);
        std::vector<float> new_gate_input_host(pres_size);

        cudaMemcpy(update_gate_input_host.data(), z_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(reset_gate_input_host.data(), r_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(new_gate_input_host.data(), g_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int t = 0; t < time_steps; ++t) {
            const float *update_gate_input_step = update_gate_input_host.data() + t * batch_size * hidden_size;
            const float *reset_gate_input_step = reset_gate_input_host.data() + t * batch_size * hidden_size;
            const float *new_gate_input_step = new_gate_input_host.data() + t * batch_size * hidden_size;

            hist_collectors.update_gate_input_hist.collect(update_gate_input_step, NH);
            hist_collectors.reset_gate_input_hist.collect(reset_gate_input_step, NH);
            hist_collectors.new_gate_input_hist.collect(new_gate_input_step, NH);
        }
    }
}

// =====================================================================
// GPU 直方图收集实现
// =====================================================================

void collectAllHistogramsGPU(
    GRUGPUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *Wx_add_bx, const float *Rh_add_br,
    const float *W, const float *R, const float *bx, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size,
    cudaStream_t stream) {
    
    const size_t x_size = time_steps * batch_size * input_size;
    const size_t h_size = time_steps * batch_size * hidden_size;
    const size_t Wx_add_bx_size = time_steps * batch_size * hidden_size * 3;
    const size_t Rh_add_br_size = time_steps * batch_size * hidden_size * 3;
    const float *h_skip_initial = h + batch_size * hidden_size;

    // 创建 streams（按需数量）
    constexpr int NUM_STREAMS = 8;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 并行收集 x, h, Wx+bx, Rh+br 的直方图
    hist_collectors.x_hist.collect(x, x_size, streams[0]);
    hist_collectors.h_hist.collect(h_skip_initial, h_size, streams[1]);
    hist_collectors.Wx_hist.collect(Wx_add_bx, Wx_add_bx_size, streams[2]);
    hist_collectors.Rh_hist.collect(Rh_add_br, Rh_add_br_size, streams[3]);

    // 并行收集 per-channel 直方图（使用零拷贝批量版本）
    if (!hist_collectors.W_batch.is_valid()) {
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.W_batch, W, input_size, streams[4]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.R_batch, R, hidden_size, streams[5]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.bx_batch, bx, 1, streams[6]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.br_batch, br, 1, streams[7]);
    }

    // 统一等待基础数据收集完成
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // 收集门值直方图（需要 h 数据完成后）
    gpu_hist::collect_gate_histograms(hist_collectors, v, h, time_steps, batch_size, hidden_size,
                                       stream);

    // 并行收集 gate input（如果提供）
    if (pres_size > 0 && z_pres && r_pres && g_pres) {
        hist_collectors.update_gate_input_hist.collect(z_pres, pres_size, streams[0]);
        hist_collectors.reset_gate_input_hist.collect(r_pres, pres_size, streams[1]);
        hist_collectors.new_gate_input_hist.collect(g_pres, pres_size, streams[2]);
        
        for (int i = 0; i < 3; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // 清理 streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

