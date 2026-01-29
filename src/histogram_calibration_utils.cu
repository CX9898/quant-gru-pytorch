// =====================================================================
// 直方图校准辅助函数实现（histogram_calibration_utils.cu）
// =====================================================================

#include "histogram_calibration_utils.h"

#include <cuda_runtime.h>

#include <vector>

#include "dev_vector.h"

// =====================================================================
// GPU 直方图收集实现
// =====================================================================

void collectAllHistogramsGPU(
    GRUGPUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *Wx_add_bw, const float *Rh_add_br,
    const float *W, const float *R, const float *bw, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size,
    cudaStream_t stream) {
    
    const size_t x_size = time_steps * batch_size * input_size;
    const size_t h_size = time_steps * batch_size * hidden_size;
    const size_t Wx_add_bw_size = time_steps * batch_size * hidden_size * 3;
    const size_t Rh_add_br_size = time_steps * batch_size * hidden_size * 3;
    const float *h_skip_initial = h + batch_size * hidden_size;

    // 创建 streams（按需数量）
    constexpr int NUM_STREAMS = 8;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 并行收集 x, h, Wx+bw, Rh+br 的直方图
    hist_collectors.x_hist.collect(x, x_size, streams[0]);
    hist_collectors.h_hist.collect(h_skip_initial, h_size, streams[1]);
    hist_collectors.Wx_hist.collect(Wx_add_bw, Wx_add_bw_size, streams[2]);
    hist_collectors.Rh_hist.collect(Rh_add_br, Rh_add_br_size, streams[3]);

    // 并行收集 per-channel 直方图（使用零拷贝批量版本）
    if (!hist_collectors.W_batch.is_valid()) {
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.W_batch, W, input_size, streams[4]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.R_batch, R, hidden_size, streams[5]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.bw_batch, bw, 1, streams[6]);
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

