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
    const float *tmp_Wx, const float *tmp_Rh,
    const float *W, const float *R, const float *bx, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size) {
    
    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;

    // 1. 收集输入 x 的直方图
    collectHistogramPerStep(hist_collectors.x_hist, x, time_steps, NI);

    // 2. 收集隐藏状态 h 的直方图（跳过初始状态）
    collectHistogramPerStep(hist_collectors.h_hist, h + NH, time_steps, NH);

    // 3. 收集 Wx 结果的直方图
    collectHistogramPerStep(hist_collectors.Wx_hist, tmp_Wx, time_steps, NH * 3);

    // 4. 收集 Rh 结果的直方图
    collectHistogramPerStep(hist_collectors.Rh_hist, tmp_Rh, time_steps, NH * 3);

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
    std::vector<float> z_out(output_size);
    std::vector<float> r_out(output_size);
    std::vector<float> g_out(output_size);
    std::vector<float> Rh_add_br_g(output_size);
    std::vector<float> rRh_g(output_size);
    std::vector<float> new_contrib(output_size);
    std::vector<float> old_contrib(output_size);

    // 解析 v 中的值
    for (int t = 0; t < time_steps; ++t) {
        for (int b = 0; b < batch_size; ++b) {
            const size_t v_base = t * batch_size * hidden_size * 4 + b * hidden_size * 4;
            const size_t out_base = t * batch_size * hidden_size + b * hidden_size;

            for (int hh = 0; hh < hidden_size; ++hh) {
                const float z_val = v_host[v_base + 0 * hidden_size + hh];
                const float r_val = v_host[v_base + 1 * hidden_size + hh];
                const float g_val = v_host[v_base + 2 * hidden_size + hh];
                const float Rh_add_br_val = v_host[v_base + 3 * hidden_size + hh];

                z_out[out_base + hh] = z_val;
                r_out[out_base + hh] = r_val;
                g_out[out_base + hh] = g_val;
                Rh_add_br_g[out_base + hh] = Rh_add_br_val;
                rRh_g[out_base + hh] = r_val * Rh_add_br_val;
                new_contrib[out_base + hh] = (1.0f - z_val) * g_val;

                // h_old 是上一个时间步的隐藏状态
                const size_t h_base = t * batch_size * hidden_size + b * hidden_size;
                old_contrib[out_base + hh] = z_val * h_host[h_base + hh];
            }
        }
    }

    // 分时间步收集直方图
    for (int t = 0; t < time_steps; ++t) {
        const float *z_step = z_out.data() + t * batch_size * hidden_size;
        const float *r_step = r_out.data() + t * batch_size * hidden_size;
        const float *g_step = g_out.data() + t * batch_size * hidden_size;
        const float *Rh_add_br_step = Rh_add_br_g.data() + t * batch_size * hidden_size;
        const float *rRh_step = rRh_g.data() + t * batch_size * hidden_size;
        const float *new_contrib_step = new_contrib.data() + t * batch_size * hidden_size;
        const float *old_contrib_step = old_contrib.data() + t * batch_size * hidden_size;

        hist_collectors.z_out_hist.collect(z_step, NH);
        hist_collectors.r_out_hist.collect(r_step, NH);
        hist_collectors.g_out_hist.collect(g_step, NH);
        hist_collectors.Rh_add_br_g_hist.collect(Rh_add_br_step, NH);
        hist_collectors.rRh_hist.collect(rRh_step, NH);
        hist_collectors.new_contrib_hist.collect(new_contrib_step, NH);
        hist_collectors.old_contrib_hist.collect(old_contrib_step, NH);
    }

    // 7. 收集 z_pre, r_pre, g_pre 的直方图（如果提供）
    if (pres_size > 0 && z_pres && r_pres && g_pres) {
        std::vector<float> z_pres_host(pres_size);
        std::vector<float> r_pres_host(pres_size);
        std::vector<float> g_pres_host(pres_size);

        cudaMemcpy(z_pres_host.data(), z_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(r_pres_host.data(), r_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(g_pres_host.data(), g_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int t = 0; t < time_steps; ++t) {
            const float *z_pre_step = z_pres_host.data() + t * batch_size * hidden_size;
            const float *r_pre_step = r_pres_host.data() + t * batch_size * hidden_size;
            const float *g_pre_step = g_pres_host.data() + t * batch_size * hidden_size;

            hist_collectors.z_pre_hist.collect(z_pre_step, NH);
            hist_collectors.r_pre_hist.collect(r_pre_step, NH);
            hist_collectors.g_pre_hist.collect(g_pre_step, NH);
        }
    }
}

// =====================================================================
// GPU 直方图收集实现
// =====================================================================

void collectAllHistogramsGPU(
    GRUGPUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *tmp_Wx, const float *tmp_Rh,
    const float *W, const float *R, const float *bx, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size,
    cudaStream_t stream) {
    
    const size_t x_size = time_steps * batch_size * input_size;
    const size_t h_size = time_steps * batch_size * hidden_size;
    const size_t Wx_size = time_steps * batch_size * hidden_size * 3;
    const size_t Rh_size = time_steps * batch_size * hidden_size * 3;
    const float *h_skip_initial = h + batch_size * hidden_size;

    // 创建 streams（按需数量）
    constexpr int NUM_STREAMS = 8;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 并行收集 x, h, Wx, Rh 的直方图
    hist_collectors.x_hist.collect(x, x_size, streams[0]);
    hist_collectors.h_hist.collect(h_skip_initial, h_size, streams[1]);
    hist_collectors.Wx_hist.collect(tmp_Wx, Wx_size, streams[2]);
    hist_collectors.Rh_hist.collect(tmp_Rh, Rh_size, streams[3]);

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

    // 并行收集 z_pre, r_pre, g_pre（如果提供）
    if (pres_size > 0 && z_pres && r_pres && g_pres) {
        hist_collectors.z_pre_hist.collect(z_pres, pres_size, streams[0]);
        hist_collectors.r_pre_hist.collect(r_pres, pres_size, streams[1]);
        hist_collectors.g_pre_hist.collect(g_pres, pres_size, streams[2]);
        
        for (int i = 0; i < 3; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // 清理 streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

