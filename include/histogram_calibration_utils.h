// =====================================================================
// 直方图校准辅助函数（histogram_calibration_utils.h）
// =====================================================================
// 提供直方图收集相关的辅助函数
// =====================================================================

#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "histogram_collector.h"
#include "calibration_gpu.cuh"
#include "quantize_bitwidth_config.h"

// =====================================================================
// GPU 直方图收集函数声明
// =====================================================================

/// 使用 GPU 收集所有直方图（高性能版本）
/// 所有直方图计算都在 GPU 上完成，避免大量 GPU->CPU 数据传输
/// 使用 CUDA streams 并行收集多个直方图
void collectAllHistogramsGPU(
    GRUGPUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *Wx_add_bw, const float *Rh_add_br,
    const float *W, const float *R, const float *bw, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size,
    const OperatorQuantConfig &bitwidth_config,
    cudaStream_t stream = 0);


