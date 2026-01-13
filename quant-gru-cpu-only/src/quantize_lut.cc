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

SigmoidLUT generate_sigmoid_lut(int8_t shift_x, int32_t zp_x, int8_t shift_y, int32_t zp_y,
                                 QuantBitWidth in_bw, QuantBitWidth out_bw) {
    float scale_x = std::pow(2.0f, -shift_x);
    float x_min = std::max((in_bw.qmin() - zp_x) * scale_x, -8.0f);
    float x_max = std::min((in_bw.qmax() - zp_x) * scale_x, 8.0f);

    SigmoidLUT lut = {};
    lut.shift_bits_x = shift_x; lut.zp_x = zp_x;
    lut.shift_bits_y = shift_y; lut.zp_y = zp_y;

    auto pts = adaptive_segmentation(x_min, x_max, NUM_SEGMENTS);
    
    struct Coef { float x_end, b, c; };
    std::vector<Coef> coeffs(NUM_SEGMENTS);
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        std::vector<float> xs(100), ys(100);
        for (int j = 0; j < 100; j++) {
            xs[j] = pts[i] + (pts[i + 1] - pts[i]) * j / 99;
            ys[j] = 1.0f / (1.0f + std::exp(-xs[j]));
        }
        linear_fit(xs, ys, coeffs[i].b, coeffs[i].c);
        coeffs[i].x_end = pts[i + 1];
    }

    float scale_y = std::pow(2.0f, -shift_y);
    float b_max = 0, c_max = 0;
    for (auto &co : coeffs) {
        b_max = std::max(b_max, std::abs(co.b));
        c_max = std::max(c_max, std::abs(co.c + zp_y * scale_y));
    }

    int8_t shift_b = (out_bw.bits_ <= 8) ? determine_shift_bits_int8(b_max) : determine_shift_bits_int16(b_max);
    int8_t shift_c = (out_bw.bits_ <= 8) ? determine_shift_bits_int8(c_max) : determine_shift_bits_int16(c_max);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        int32_t q_b = quantize_coefficient_int32(coeffs[i].b, shift_b);
        int32_t q_c = quantize_coefficient_int32(coeffs[i].c + zp_y * scale_y, shift_c);
        int8_t n_bx = shift_b + shift_x - shift_y;
        int8_t n_yc = shift_c - shift_y;
        
        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_bx;
        lut.segments[i].term_c_precomputed = (n_yc >= 0) ? (q_c >> n_yc) : (q_c << -n_yc);
        lut.segments[i].threshold = clamp_by_bitwidth(static_cast<int32_t>(std::round(coeffs[i].x_end / scale_x + zp_x)), in_bw);
    }
    return lut;
}

SigmoidLUT generate_tanh_lut(int8_t shift_x, int32_t zp_x, int8_t shift_y, int32_t zp_y,
                              QuantBitWidth in_bw, QuantBitWidth out_bw) {
    float scale_x = std::pow(2.0f, -shift_x);
    float x_min = std::max((in_bw.qmin() - zp_x) * scale_x, -4.0f);
    float x_max = std::min((in_bw.qmax() - zp_x) * scale_x, 4.0f);

    SigmoidLUT lut = {};
    lut.shift_bits_x = shift_x; lut.zp_x = zp_x;
    lut.shift_bits_y = shift_y; lut.zp_y = zp_y;

    auto pts = adaptive_segmentation(x_min, x_max, NUM_SEGMENTS);
    
    struct Coef { float x_end, b, c; };
    std::vector<Coef> coeffs(NUM_SEGMENTS);
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        std::vector<float> xs(100), ys(100);
        for (int j = 0; j < 100; j++) {
            xs[j] = pts[i] + (pts[i + 1] - pts[i]) * j / 99;
            ys[j] = std::tanh(xs[j]);
        }
        linear_fit(xs, ys, coeffs[i].b, coeffs[i].c);
        coeffs[i].x_end = pts[i + 1];
    }

    float scale_y = std::pow(2.0f, -shift_y);
    float b_max = 0, c_max = 0;
    for (auto &co : coeffs) {
        b_max = std::max(b_max, std::abs(co.b));
        c_max = std::max(c_max, std::abs(co.c + zp_y * scale_y));
    }

    int8_t shift_b = (out_bw.bits_ <= 8) ? determine_shift_bits_int8(b_max) : determine_shift_bits_int16(b_max);
    int8_t shift_c = (out_bw.bits_ <= 8) ? determine_shift_bits_int8(c_max) : determine_shift_bits_int16(c_max);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        int32_t q_b = quantize_coefficient_int32(coeffs[i].b, shift_b);
        int32_t q_c = quantize_coefficient_int32(coeffs[i].c + zp_y * scale_y, shift_c);
        int8_t n_bx = shift_b + shift_x - shift_y;
        int8_t n_yc = shift_c - shift_y;
        
        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_bx;
        lut.segments[i].term_c_precomputed = (n_yc >= 0) ? (q_c >> n_yc) : (q_c << -n_yc);
        lut.segments[i].threshold = clamp_by_bitwidth(static_cast<int32_t>(std::round(coeffs[i].x_end / scale_x + zp_x)), in_bw);
    }
    return lut;
}
