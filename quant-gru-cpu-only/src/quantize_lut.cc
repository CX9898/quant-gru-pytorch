#include "quantize_lut_types.h"
#include "quantize_ops_helper.h"
#include <algorithm>
#include <cmath>
#include <vector>

static void linear_fit(const std::vector<float> &x, const std::vector<float> &y, float &b, float &c) {
    int n = static_cast<int>(x.size());
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < n; i++) {
        sum_x += x[i]; sum_y += y[i]; sum_xy += x[i] * y[i]; sum_x2 += x[i] * x[i];
    }
    float denom = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-9f) { b = 0; c = sum_y / n; return; }
    b = (n * sum_xy - sum_x * sum_y) / denom;
    c = (sum_y - b * sum_x) / n;
}

static std::vector<float> adaptive_segmentation(float x_min, float x_max, int num_segments) {
    const int N = 1000;
    std::vector<float> xs(N), ws(N - 1);
    for (int i = 0; i < N; i++) xs[i] = x_min + (x_max - x_min) * i / (N - 1);
    
    for (int i = 0; i < N - 1; i++) {
        float y = 1.0f / (1.0f + std::exp(-xs[i]));
        float y_next = 1.0f / (1.0f + std::exp(-xs[i + 1]));
        float slope = std::abs(y_next - y) / (xs[i + 1] - xs[i] + 1e-9f);
        float dist = std::abs(xs[i]);
        ws[i] = (dist < 2.0f) ? 5.0f * (1.0f - dist / 2.0f) + 1.0f : 1.0f + slope * 0.5f;
    }
    
    float sum = 0;
    for (auto w : ws) sum += w;
    for (auto &w : ws) w /= sum;
    
    std::vector<float> cum(N - 1);
    cum[0] = ws[0];
    for (int i = 1; i < N - 1; i++) cum[i] = cum[i - 1] + ws[i];
    
    std::vector<float> pts = {x_min};
    for (int i = 1; i < num_segments; i++) {
        float target = static_cast<float>(i) / num_segments;
        auto it = std::lower_bound(cum.begin(), cum.end(), target);
        int idx = std::min(static_cast<int>(it - cum.begin()), N - 2);
        pts.push_back(xs[idx]);
    }
    pts.push_back(x_max);
    return pts;
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
