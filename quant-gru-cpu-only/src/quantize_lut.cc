#include "quantize_lut_types.h"
#include "quantize_ops_helper.h"
#include <algorithm>
#include <cmath>
#include <vector>

// 线性拟合函数（最小二乘法）
// 与 src/quantize_ops.cu 保持一致
static void linear_fit(const std::vector<float> &x, const std::vector<float> &y, float &b,
                       float &c) {
    int n = x.size();
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    float denom = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-9f) {
        b = 0.0f;
        c = sum_y / n;
        return;
    }

    b = (n * sum_xy - sum_x * sum_y) / denom;
    c = (sum_y - b * sum_x) / n;
}

// 自适应分段（Sigmoid/Tanh 专用）
// 与 src/quantize_ops.cu 中的 adaptive_segmentation_sigmoid 保持一致
static std::vector<float> adaptive_segmentation(float x_min, float x_max, int num_segments) {
    // Sigmoid/Tanh 的权重配置
    const float centerWeight = 5.0f;
    const float centerRange = 2.0f;

    const int numSamples = 1000;
    std::vector<float> xSamples(numSamples);
    std::vector<float> weights(numSamples - 1);

    for (int i = 0; i < numSamples; i++) {
        xSamples[i] = x_min + (x_max - x_min) * static_cast<float>(i) / (numSamples - 1);
    }

    for (int i = 0; i < numSamples - 1; i++) {
        float x = xSamples[i];
        float x_next = xSamples[i + 1];

        float y = 1.0f / (1.0f + std::exp(-x));
        float y_next = 1.0f / (1.0f + std::exp(-x_next));
        float slope = std::abs(y_next - y) / (x_next - x + 1e-9f);

        float distToCenter = std::abs(x);

        if (distToCenter < centerRange) {
            weights[i] = centerWeight * (1.0f - distToCenter / centerRange) + 1.0f;
        } else {
            weights[i] = 1.0f + slope * 0.5f;
        }
    }

    float sumWeights = 0.0f;
    for (int i = 0; i < numSamples - 1; i++) {
        sumWeights += weights[i];
    }
    for (int i = 0; i < numSamples - 1; i++) {
        weights[i] /= sumWeights;
    }

    std::vector<float> cumWeights(numSamples - 1);
    cumWeights[0] = weights[0];
    for (int i = 1; i < numSamples - 1; i++) {
        cumWeights[i] = cumWeights[i - 1] + weights[i];
    }

    std::vector<float> points;
    points.push_back(x_min);

    for (int i = 1; i < num_segments; i++) {
        float target = static_cast<float>(i) / num_segments;
        auto it = std::lower_bound(cumWeights.begin(), cumWeights.end(), target);
        int idx = static_cast<int>(std::distance(cumWeights.begin(), it));
        if (idx >= numSamples - 1) idx = numSamples - 2;
        if (idx < 0) idx = 0;
        points.push_back(xSamples[idx]);
    }

    points.push_back(x_max);

    // 确保点单调递增且无重复
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end(),
                            [](float a, float b) { return std::abs(a - b) < 1e-9f; });
    points.erase(last, points.end());

    // 如果去重后点数不够，在最大间隔处插入点
    while (static_cast<int>(points.size()) < num_segments + 1) {
        float max_gap = 0.0f;
        size_t max_gap_idx = 0;
        for (size_t i = 0; i < points.size() - 1; i++) {
            float gap = points[i + 1] - points[i];
            if (gap > max_gap) {
                max_gap = gap;
                max_gap_idx = i;
            }
        }
        float new_point = (points[max_gap_idx] + points[max_gap_idx + 1]) / 2.0f;
        points.insert(points.begin() + max_gap_idx + 1, new_point);
    }

    return points;
}

SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                 int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    // 根据输入位宽确定量化范围（任意位宽支持，与主项目一致）
    int32_t quant_min = input_bw.qmin();
    int32_t quant_max = input_bw.qmax();

    float scale_x = exp2_scale(shift_bits_x);
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Sigmoid 有效范围限制
    constexpr float SIGMOID_EFFECTIVE_RANGE = 8.0f;
    x_min = std::max(x_min, -SIGMOID_EFFECTIVE_RANGE);
    x_max = std::min(x_max, SIGMOID_EFFECTIVE_RANGE);

    SigmoidLUT lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 生成分段点
    std::vector<float> segment_points = adaptive_segmentation(x_min, x_max, NUM_SEGMENTS);

    // 第一遍扫描：拟合所有分段
    struct SegmentCoeffs { float x_start, x_end, b, c; };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples), y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));  // Sigmoid
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    // 第二遍扫描：统一量化参数
    float scale_y = exp2_scale(shift_bits_y);
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    float b_abs_max = 0.0f, c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    // 根据输出位宽自动确定 shift_bits（与主项目一致）
    int8_t shift_bits_b = determine_shift_bits(b_abs_max, output_bw);
    int8_t shift_bits_c = determine_shift_bits(c_abs_max, output_bw);

    // 第三遍扫描：量化每段
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        int32_t q_b = quantize_coefficient_int32(coeff.b, shift_bits_b);
        int32_t q_c = quantize_coefficient_int32(c_adjusted, shift_bits_c);

        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int32_t term_c_precomputed = (n_yc >= 0) ? (q_c >> n_yc) : (q_c << (-n_yc));

        // threshold 量化（任意位宽支持，存储为 int32_t，与主项目一致）
        int32_t threshold = round_to_int(coeff.x_end / scale_x + zp_x);
        threshold = clamp_by_bitwidth(threshold, input_bw);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                              int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    // 根据输入位宽确定量化范围（任意位宽支持，与主项目一致）
    int32_t quant_min = input_bw.qmin();
    int32_t quant_max = input_bw.qmax();

    float scale_x = exp2_scale(shift_bits_x);
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Tanh 有效范围限制
    constexpr float TANH_EFFECTIVE_RANGE = 4.0f;
    x_min = std::max(x_min, -TANH_EFFECTIVE_RANGE);
    x_max = std::min(x_max, TANH_EFFECTIVE_RANGE);

    SigmoidLUT lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    std::vector<float> segment_points = adaptive_segmentation(x_min, x_max, NUM_SEGMENTS);

    struct SegmentCoeffs { float x_start, x_end, b, c; };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples), y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);  // Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    float scale_y = exp2_scale(shift_bits_y);
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    float b_abs_max = 0.0f, c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    // 根据输出位宽自动确定 shift_bits（与主项目一致）
    int8_t shift_bits_b = determine_shift_bits(b_abs_max, output_bw);
    int8_t shift_bits_c = determine_shift_bits(c_abs_max, output_bw);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        int32_t q_b = quantize_coefficient_int32(coeff.b, shift_bits_b);
        int32_t q_c = quantize_coefficient_int32(c_adjusted, shift_bits_c);

        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int32_t term_c_precomputed = (n_yc >= 0) ? (q_c >> n_yc) : (q_c << (-n_yc));
        
        // threshold 量化（任意位宽支持，存储为 int32_t，与主项目一致）
        int32_t threshold = round_to_int(coeff.x_end / scale_x + zp_x);
        threshold = clamp_by_bitwidth(threshold, input_bw);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}
