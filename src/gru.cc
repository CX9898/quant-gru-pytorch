#include <Eigen/Dense>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "devVector.h"
#include "device_ptr.h"
#include "gru.h"
#include "gru_quant.h"

using Tensor1f = Eigen::Tensor<float, 1>;
using Tensor2f = Eigen::Tensor<float, 2>;
using Tensor3f = Eigen::Tensor<float, 3>;
using Tensor1i8 = Eigen::Tensor<int8_t, 1>;
using Tensor2i8 = Eigen::Tensor<int8_t, 2>;
using Tensor3i8 = Eigen::Tensor<int8_t, 3>;
using Tensor1i16 = Eigen::Tensor<int16_t, 1>;
using Tensor2i16 = Eigen::Tensor<int16_t, 2>;
using Tensor3i16 = Eigen::Tensor<int16_t, 3>;
using Tensor1i32 = Eigen::Tensor<int32_t, 1>;
using Tensor2i32 = Eigen::Tensor<int32_t, 2>;
using Tensor3i32 = Eigen::Tensor<int32_t, 3>;

constexpr int BATCH_SIZE = 64;     // 批大小
constexpr int SEQUENCE_LEN = 1000; // 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 512; // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 512;  // 输入维度(I), x_t的维度

cublasHandle_t g_blas_handle;  // 改为非static以便在wrapper中访问

// 初始化函数，供Python绑定调用
void init_gru_cublas() {
    if (g_blas_handle == nullptr) {
        cublasCreate(&g_blas_handle);
    }
}

class ScopeTimer { // 测量时间类
 public:
  ScopeTimer(const std::string &msg) : msg_(msg) {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
      cudaDeviceSynchronize();
      cudaEventRecord(start_);
  }

  ~ScopeTimer() {
      float elapsed_ms;
      cudaEventRecord(stop_);
      cudaEventSynchronize(stop_);
      cudaEventElapsedTime(&elapsed_ms, start_, stop_);
      printf("%s %fms\n", msg_.c_str(), elapsed_ms);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
  }

 private:
  std::string msg_;
  cudaEvent_t start_, stop_;
};

template<bool use_int16 = false>
// 控制量化精度位宽
void GruQuantInit(
    const Tensor2f &W, // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const Tensor2f &R,  // 隐藏层到隐藏层的循环权重矩阵
    const Tensor1f &bx, // 输入偏置项（input bias），来自输入路径
    const Tensor1f &br, // 循环偏置项（recurrent bias），来自循环路径
    const Tensor3f &x, // 输入序列张量
    const Tensor3f &dh_new, // 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    Tensor2i8 &W_quant,
    Tensor2i8 &R_quant,
    Tensor1i32 &bx_quant,
    Tensor1i32 &br_quant,
    Tensor3i8 &x_quant,
    Tensor3i8 &dh_new_quant,
    const GRUQuantitativeParameters &gruRescaleParams
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // N : batch_size
    // C : input_size
    if (!use_int16) { // int8量化
        // 权重是per-channel的，大小为H * 3（hidden_size * 3）
        // W: [H*3, C]，W_quant: [H*3, C]，scale_W_: [H*3]
        for (int i = 0; i < W.dimension(0); ++i) {  // i: [0, H*3)
            float scale_W = gruRescaleParams.scale_W_[i];
            for (int j = 0; j < W.dimension(1); ++j) {  // j: [0, input_size)
                float real = W(i, j);
                // 对称量化到int8：clip到[-128,127]
                int32_t q = static_cast<int32_t>(std::round(real / scale_W));
                q = std::max(-128, std::min(127, q));
                W_quant(i, j) = static_cast<int8_t>(q);
            }
        }
        // R: [H*3, H]，R_quant: [H*3, H]，scale_R_: [H*3]
        for (int i = 0; i < R.dimension(0); ++i) {  // i: [0, H*3)
            float scale_R = gruRescaleParams.scale_R_[i];
            for (int j = 0; j < R.dimension(1); ++j) {  // j: [0, hidden_size)
                float real = R(i, j);
                int32_t q = static_cast<int32_t>(std::round(real / scale_R));
                q = std::max(-128, std::min(127, q));
                R_quant(i, j) = static_cast<int8_t>(q);
            }
        }

        // 偏置per-channel，H*3
        // bx_quant: [H*3], scale_bx_: [H*3]
        for (int i = 0; i < bx.dimension(0); ++i) {  // i: [0, H*3)
            float scale_bx = gruRescaleParams.scale_bx_[i];
            float real = bx(i);
            int32_t q = static_cast<int32_t>(std::round(real / scale_bx));
            bx_quant(i) = q;
        }
        // br_quant: [H*3], scale_br_: [H*3]
        for (int i = 0; i < br.dimension(0); ++i) {  // i: [0, H*3)
            float scale_br = gruRescaleParams.scale_br_[i];
            float real = br(i);
            int32_t q = static_cast<int32_t>(std::round(real / scale_br));
            br_quant(i) = q;
        }

        // x: [C, N, T], x_quant: [C, N, T]
        // 量化用全局scale_x_和zp_x_
        for (int t = 0; t < x.dimension(2); ++t) {      // t: [0, time_steps)
            for (int n = 0; n < x.dimension(1); ++n) {  // n: [0, batch_size)
                for (int c = 0; c < x.dimension(0); ++c) {  // c: [0, input_size)
                    float real = x(c, n, t);
                    int32_t q =
                        static_cast<int32_t>(std::round(real / gruRescaleParams.scale_x_)) + gruRescaleParams.zp_x_;
                    q = std::max(-128, std::min(127, q));
                    x_quant(c, n, t) = static_cast<int8_t>(q);
                }
            }
        }

        // dh_new: [H, N, T+1], dh_new_quant: [H, N, T+1]
        // 使用scale_h_和zp_h_
        for (int t = 0; t < dh_new.dimension(2); ++t) {    // t: [0, time_steps+1)
            for (int n = 0; n < dh_new.dimension(1); ++n) { // n: [0, batch_size)
                for (int h = 0; h < dh_new.dimension(0); ++h) { // h: [0, hidden_size)
                    float real = dh_new(h, n, t);
                    int32_t q =
                        static_cast<int32_t>(std::round(real / gruRescaleParams.scale_h_)) + gruRescaleParams.zp_h_;
                    q = std::max(-128, std::min(127, q));
                    dh_new_quant(h, n, t) = static_cast<int8_t>(q);
                }
            }
        }


    } else {
        // int16量化
    }


}

template<bool use_int16_quant = false>
void GruInferenceQuant(const Tensor2i8 &W,
                       const Tensor2i8 &R,
                       const Tensor1i32 &bx,
                       const Tensor1i32 &br,
                       const Tensor3i8 &x,
                       const GRUQuantitativeParameters &quant_parms,
                       Tensor3i8 &h // (time_steps + 1) * batch_size * hidden_size
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    generate_int8_lut(quant_parms.scale_z_pre_, quant_parms.zp_z_pre_, quant_parms.scale_z_out_, quant_parms.zp_z_out_,
                      quant_parms.scale_r_pre_, quant_parms.zp_r_pre_, quant_parms.scale_r_out_, quant_parms.zp_r_out_,
                      quant_parms.scale_g_pre_, quant_parms.zp_g_pre_, quant_parms.scale_g_out_, quant_parms.zp_g_out_);

    // Copy weights over to GPU.
    device_ptr<Tensor2i8> W_dev(W);
    device_ptr<Tensor2i8> R_dev(R);
    device_ptr<Tensor1i32> bx_dev(bx);
    device_ptr<Tensor1i32> br_dev(br);
    device_ptr<Tensor3i8> x_dev(x);

    device_ptr<Tensor3i32> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3); // 用于存放W * x的中间结果
    device_ptr<Tensor2i32> tmp_Rh_dev(batch_size * hidden_size * 3); // 用于存放R * h的中间结果

    device_ptr<Tensor3i8> h_dev(h);
    h_dev.zero(); // h初始化为0

    {
        gru::ForwardPassQuant<int8_t> forward = gru::ForwardPassQuant<int8_t>(
            false, // training
            batch_size, input_size, hidden_size, g_blas_handle);

        // 得到量化GRU中使用的rescale参数
        forward.setRescaleParam(quant_parms);

        ScopeTimer t("Inference Quant:");
        forward.Run(time_steps, W_dev.data, R_dev.data, bx_dev.data, br_dev.data,
                    x_dev.data, h_dev.data, nullptr, tmp_Wx_dev.data, tmp_Rh_dev.data,
                    0.0f, nullptr);
    }

    h_dev.ToHost(h);
}

void GruInference(const Tensor2f &W,
                  const Tensor2f &R,
                  const Tensor1f &bx,
                  const Tensor1f &br,
                  const Tensor3f &x,
                  Tensor3f &h) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2f> W_dev(W);
    device_ptr<Tensor2f> R_dev(R);
    device_ptr<Tensor1f> bx_dev(bx);
    device_ptr<Tensor1f> br_dev(br);
    device_ptr<Tensor3f> x_dev(x);

    device_ptr<Tensor3f> h_dev(hidden_size * batch_size * (time_steps + 1));
    device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3); // 用于存放W * x的中间结果
    device_ptr<Tensor2f> tmp_Rh_dev(batch_size * hidden_size * 3); // 用于存放R * h的中间结果

    h_dev.zero(); // h初始化为0

    {
        ScopeTimer t("Inference:");

        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            false, // training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.Run(time_steps, W_dev.data, R_dev.data, bx_dev.data, br_dev.data,
                    x_dev.data, h_dev.data, nullptr, tmp_Wx_dev.data, tmp_Rh_dev.data,
                    0.0f, nullptr);
    }

    h_dev.ToHost(h);
}

void GruTrain(const Tensor2f &W, // 输入到隐藏层的权重矩阵. [input_size,
    // hidden_size * 3] 对应三个门
              const Tensor2f &R, // 隐藏层到隐藏层的循环权重矩阵
              const Tensor1f &bx, // 输入偏置项（input bias），来自输入路径
              const Tensor1f &br, // 循环偏置项（recurrent bias），来自循环路径
              const Tensor3f &x, // 输入序列张量
              const Tensor3f &dh_new, // 来自上层网络或损失函数的反向梯度.
    // [hidden_size, batch_size, time_steps]
              bool enable_quantitative = false, // 是否启用量化推理模式
              bool use_int16 = false            // 控制量化精度位宽
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2f> W_dev(W);
    device_ptr<Tensor2f> R_dev(R);
    device_ptr<Tensor1f> bx_dev(bx);
    device_ptr<Tensor1f> br_dev(br);
    device_ptr<Tensor3f> x_dev(x);
    device_ptr<Tensor3f> dh_new_dev(dh_new);

    device_ptr<Tensor2f> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor2f> tmp_Rh_dev(batch_size * hidden_size * 3);
    device_ptr<Tensor3f> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    {
        ScopeTimer t("Train forward:");
        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            true,  // training
            batch_size,
            input_size,
            hidden_size,
            g_blas_handle);

        forward.Run(
            time_steps,
            W_dev.data,
            R_dev.data,
            bx_dev.data,
            br_dev.data,
            x_dev.data,
            h_dev.data,
            v_dev.data,
            tmp_Wx_dev.data,
            tmp_Rh_dev.data,
            0.0f,
            nullptr);
    }

    device_ptr<Tensor3f> dx_dev(time_steps * batch_size *
                                input_size); // 输入序列梯度
    device_ptr<Tensor2f> dW_dev(input_size * hidden_size *
                                3); // 对输入权重的梯度
    device_ptr<Tensor2f> dR_dev(hidden_size * hidden_size *
                                3);                // 对循环权重的梯度
    device_ptr<Tensor1f> dbx_dev(hidden_size * 3); // 对输入偏置的梯度
    device_ptr<Tensor1f> dbr_dev(hidden_size * 3); // 对循环偏置的梯度
    device_ptr<Tensor2f> dh_dev(batch_size *
                                hidden_size); // 对最后隐藏状态的梯度
    device_ptr<Tensor3f> dp_dev(time_steps * batch_size * hidden_size *
                                3); // 临时缓存梯度（内部结构用）
    device_ptr<Tensor3f> dq_dev(time_steps * batch_size * hidden_size * 3); // 临时缓存梯度（内部结构用）

    {
        ScopeTimer t("Train backward:");
        gru::BackwardPass<float> backward(batch_size, input_size, hidden_size,
                                          g_blas_handle);

        backward.Run(time_steps, W_dev.data, R_dev.data, bx_dev.data, br_dev.data,
                     x_dev.data, h_dev.data, v_dev.data, dh_new_dev.data,
                     dx_dev.data, dW_dev.data, dR_dev.data, dbx_dev.data,
                     dbr_dev.data, dh_dev.data, dp_dev.data, dq_dev.data,
                     nullptr);
    }

}

// 计算余弦相似度
float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f); // 防止除零
}

void checkHQuantizationWithCosine(
    const std::vector<float> &h_inference,          // 浮点 h, size = (time_steps+1) * batch_size * hidden_size
    const std::vector<int8_t> &h_quant_inference,  // 量化 h, size 同上
    int time_steps,
    int batch_size,
    int hidden_size,
    const GRUQuantitativeParameters &scaleParam,
    float threshold = -1.0f                         // 超阈值，如果 < 0 则自动计算
) {
    // 计算阈值：量化误差的理论最大值是 scale / 2（四舍五入误差）
    // 使用 2 * scale 作为阈值，允许一定的误差范围
    if (threshold < 0.0f) {
        threshold = 2.0f * scaleParam.scale_h_;
    }

    const int size_per_step = batch_size * hidden_size;

    // 验证输入数据大小
    if (h_inference.size() != static_cast<size_t>((time_steps + 1) * size_per_step)) {
        printf("[Error] h_inference size mismatch: expected %d, got %zu\n",
               (time_steps + 1) * size_per_step, h_inference.size());
        return;
    }
    if (h_quant_inference.size() != static_cast<size_t>((time_steps + 1) * size_per_step)) {
        printf("[Error] h_quant_inference size mismatch: expected %d, got %zu\n",
               (time_steps + 1) * size_per_step, h_quant_inference.size());
        return;
    }

    printf("checkHQuantizationWithCosine: time_steps=%d, batch_size=%d, hidden_size=%d\n",
           time_steps, batch_size, hidden_size);
    printf("  scale_h_=%f, zp_h_=%d, threshold=%f\n",
           scaleParam.scale_h_, scaleParam.zp_h_, threshold);

    // 检查前几个数据点的值
    printf("  Sample data check:\n");
    printf("    h_inference size: %zu, expected: %d\n", h_inference.size(), (time_steps + 1) * size_per_step);
    printf("    h_quant_inference size: %zu, expected: %d\n",
           h_quant_inference.size(),
           (time_steps + 1) * size_per_step);

    // 检查初始状态（t=0）的数据
    printf("    h_inference[t=0, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%f ", h_inference[i]);
    }
    printf("\n");
    printf("    h_quant_inference[t=0, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%d ", static_cast<int>(h_quant_inference[i]));
    }
    printf("\n");

    // 检查第一个时间步（t=1）的数据
    const int t1_offset = size_per_step;
    printf("    h_inference[t=1, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%f ", h_inference[t1_offset + i]);
    }
    printf("\n");
    printf("    h_quant_inference[t=1, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%d ", static_cast<int>(h_quant_inference[t1_offset + i]));
    }
    printf("\n");

    // 检查是否有非零值
    int non_zero_float = 0, non_zero_quant = 0;
    for (size_t i = 0; i < h_inference.size(); ++i) {
        if (std::abs(h_inference[i]) > 1e-6f) non_zero_float++;
    }
    for (size_t i = 0; i < h_quant_inference.size(); ++i) {
        if (h_quant_inference[i] != 0) non_zero_quant++;
    }
    printf("    Non-zero values: h_inference=%d/%zu, h_quant_inference=%d/%zu\n",
           non_zero_float, h_inference.size(), non_zero_quant, h_quant_inference.size());

    std::vector<float> h_float_step(size_per_step);
    std::vector<float> h_quant_step(size_per_step);

    for (int t = 1; t <= time_steps; ++t) {
        // ForwardPass 存储 h 的方式：h + t * (batch_size * hidden_size)
        // 每个时间步内部：按 [batch0_h0, batch0_h1, ..., batch0_hH-1, batch1_h0, ..., batchN-1_hH-1] 顺序
        // 即：t * (N*H) + n * H + h

        const size_t t_offset = static_cast<size_t>(t) * size_per_step;

        // 直接拷贝和反量化
        for (int idx = 0; idx < size_per_step; ++idx) {
            h_float_step[idx] = h_inference[t_offset + idx];

            // 反量化: dequant = (quant - zp) * scale
            int8_t quant_val = h_quant_inference[t_offset + idx];
            h_quant_step[idx] = static_cast<float>(quant_val - scaleParam.zp_h_) * scaleParam.scale_h_;
            if ((quant_val == 0 || h_quant_step[idx] == 0) && h_float_step[idx] != 0) {
                printf("Error!, quant_val = %d, h_quant_step[%d] = %f\n", quant_val, idx, h_quant_step[idx]);
                return;
            }
        }

        // 差值统计
        float max_diff = 0.0f;
        float sum_diff = 0.0f;

        int count = 0;
        for (int idx = 0; idx < size_per_step; ++idx) {
            float diff = std::abs(h_float_step[idx] - h_quant_step[idx]);
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;

            if (diff > threshold) {
                count++;
                if (count < 5) {
                    printf("[Warning] t=%d idx=%d diff=%f h_float=%f h_quant=%f\n",
                           t, idx, diff, h_float_step[idx], h_quant_step[idx]);
                }
            }
        }
        const float baifenbi = static_cast<float>(count) / static_cast<float>(size_per_step);

        float mean_diff = sum_diff / size_per_step;
        float cos_sim = cosineSimilarity(h_float_step, h_quant_step);

        printf("Time step %d: max_diff=%f, mean_diff=%f, cosine_sim=%f, baifenbi = %f\n",
               t, max_diff, mean_diff, cos_sim, baifenbi);
    }
}

template<typename QuantT>
void calibrateGruScales(const Tensor2f &W,
                        const Tensor2f &R,
                        const Tensor1f &bx,
                        const Tensor1f &br,
                        const Tensor3f &x,
                        GRUQuantitativeParameters &quant_gru_scales
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2f> W_dev(W);
    device_ptr<Tensor2f> R_dev(R);
    device_ptr<Tensor1f> bx_dev(bx);
    device_ptr<Tensor1f> br_dev(br);
    device_ptr<Tensor3f> x_dev(x);
//    device_ptr<Tensor3f> dh_new_dev(dh_new);

    device_ptr<Tensor2f> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor2f> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor3f> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        true,  // training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.setCalibrationMode(true, false);

    forward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        v_dev.data,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        0.0f,
        nullptr);

    quant_gru_scales = forward.getGRUQuantitativeParameters();

}

void printGRUQuantitativeParameters(const GRUQuantitativeParameters &quant_gru_scales) {
    printf("GRUQuantitativeParameters (量化参数):\n");
    printf("  scale_x_ = %f\n",
           quant_gru_scales.scale_x_);
    printf("  zp_x_    = %d\n", quant_gru_scales.zp_x_);
    printf("  scale_h_ = %f\n",
           quant_gru_scales.scale_h_);
    printf("  zp_h_    = %d\n", quant_gru_scales.zp_h_);

    printf("  scale_W_ (size %zu): ", quant_gru_scales.scale_W_.size());
    for (size_t i = 0; i < quant_gru_scales.scale_W_.size() && i < 8; ++i) {
        printf("%f ", quant_gru_scales.scale_W_[i]);
    }
    if (quant_gru_scales.scale_W_.size() > 8) printf("...");
    printf("\n");

    printf("  scale_R_ (size %zu): ", quant_gru_scales.scale_R_.size());
    for (size_t i = 0; i < quant_gru_scales.scale_R_.size() && i < 8; ++i) {
        printf("%f ", quant_gru_scales.scale_R_[i]);
    }
    if (quant_gru_scales.scale_R_.size() > 8) printf("...");
    printf("\n");

    printf("  scale_bx_ (size %zu): ", quant_gru_scales.scale_bx_.size());
    for (size_t i = 0; i < quant_gru_scales.scale_bx_.size() && i < 8; ++i) {
        printf("%f ", quant_gru_scales.scale_bx_[i]);
    }
    if (quant_gru_scales.scale_bx_.size() > 8) printf("...");
    printf("\n");

    printf("  scale_br_ (size %zu): ", quant_gru_scales.scale_br_.size());
    for (size_t i = 0; i < quant_gru_scales.scale_br_.size() && i < 8; ++i) {
        printf("%f ", quant_gru_scales.scale_br_[i]);
    }
    if (quant_gru_scales.scale_br_.size() > 8) printf("...");
    printf("\n");

    printf("  scale_Wx_ = %f, zp_Wx_ = %d \n",
           quant_gru_scales.scale_Wx_, quant_gru_scales.zp_Wx_);
    printf("  scale_Rh_ = %f, zp_Rh_ = %d \n",
           quant_gru_scales.scale_Rh_, quant_gru_scales.zp_Rh_);
    printf("  scale_z_pre_ = %f, zp_z_pre_ = %d \n",
           quant_gru_scales.scale_z_pre_, quant_gru_scales.zp_z_pre_);
    printf("  scale_r_pre_ = %f, zp_r_pre_ = %d\n",
           quant_gru_scales.scale_r_pre_, quant_gru_scales.zp_r_pre_);
    printf("  scale_g_pre_ = %f, zp_g_pre_ = %d\n",
           quant_gru_scales.scale_g_pre_, quant_gru_scales.zp_g_pre_);
    printf("  scale_one_minus_update_ = %f, zp_one_minus_update_ = %d\n",
           quant_gru_scales.scale_one_minus_update_,
           quant_gru_scales.zp_one_minus_update_);
    printf("  scale_new_contrib_ = %f, zp_new_contrib_ = %d\n",
           quant_gru_scales.scale_new_contrib_,
           quant_gru_scales.zp_new_contrib_);
    printf("  scale_old_contrib_ = %f, zp_old_contrib_ = %d\n",
           quant_gru_scales.scale_old_contrib_,
           quant_gru_scales.zp_old_contrib_);
    printf("  hidden_ = %d\n", quant_gru_scales.hidden_);
}

void checkQuant(const Tensor2i8 &W_quant,  // 对应W_z/W_r/W_h的合并
                const Tensor2i8 &R_quant, // 对应R_z/R_r/R_h的合并
                const Tensor1i32 &bx_quant, // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
                const Tensor1i32 &br_quant, // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给“隐藏状态 h_{t-1} 到门控的线性变换” 加偏置
                const Tensor3i8 &x_quant,
                const Tensor3i8 &dh_new_quant
) {
    for (int i = 0; i < W_quant.size(); ++i) {
        if (W_quant.data()[i] == 0) {
            printf("Error, W_quant[%d] = %d\n", i, W_quant.data()[i]);
            break;
        }
    }
    for (int i = 0; i < R_quant.size(); ++i) {
        if (R_quant.data()[i] == 0) {
            printf("Error, R_quant[%d] = %d\n", i, R_quant.data()[i]);
            break;
        }
    }
    for (int i = 0; i < bx_quant.size(); ++i) {
        if (bx_quant.data()[i] == 0) {
            printf("Error, bx_quant[%d] = %d\n", i, bx_quant.data()[i]);
            break;
        }
    }
    for (int i = 0; i < br_quant.size(); ++i) {
        if (br_quant.data()[i] == 0) {
            printf("Error, br_quant[%d] = %d\n", i, br_quant.data()[i]);
            break;
        }
    }
    for (int i = 0; i < x_quant.size(); ++i) {
        if (x_quant.data()[i] == 0) {
            printf("Error, x_quant[%d] = %d\n", i, x_quant.data()[i]);
            break;
        }
    }
    for (int i = 0; i < dh_new_quant.size(); ++i) {
        if (dh_new_quant.data()[i] == 0) {
            printf("Error, dh_new_quant[%d] = %d", i, dh_new_quant.data()[i]);
            break;
        }
    }
    printf("Quant values check over\n");
}

bool checkScale(const float *src, size_t size, const int8_t *quant, float scale, int32_t zero_point) {
    // 计算阈值：量化误差的理论最大值是 scale / 2（四舍五入误差）
    // 使用 2 * scale 作为阈值，允许一定的误差范围
    float threshold = 2.0f * scale;
    // 确保阈值至少为 1e-6f，避免 scale 过小时阈值过小
    threshold = std::max(threshold, 1e-6f);
    for (int i = 0; i < size; ++i) {
        const float val = src[i];
        const float req_val = (quant[i] - zero_point) * scale;
        const float diff = std::abs(val - req_val);

        if (diff > threshold) {
            printf("Error, src[%d] = %f, req_val[%d] = %f, diff = %f, threshold = %f, scale = %f\n", i, val, i, req_val, diff, threshold, scale);
            return false;
        }
    }
    return true;
}

template<typename QuantT>
bool checkScalePerChannel(const float *src, size_t channel_size, size_t in_dim, const QuantT *quant, std::vector<float> scale) {
    for (int i = 0; i < in_dim; ++i) {
        for(int j = 0; j < channel_size; ++j) {
            const float val = src[i * channel_size + j];
            const float req_val = (quant[i * channel_size + j]) * scale[j];
            const float diff = std::abs(val - req_val);
            // 计算阈值：量化误差的理论最大值是 scale / 2（四舍五入误差）
            // 使用 2 * scale 作为阈值，允许一定的误差范围
            float threshold = 2.0f * scale[j];
            // 确保阈值至少为 1e-6f，避免 scale 过小时阈值过小
            threshold = std::max(threshold, 1e-6f);
            if (diff > threshold) {
                printf("Error, src[%d][%d] = %f, req_val[%d][%d] = %f, diff = %f, threshold = %f, scale = %f\n", i, j, val, i, j, req_val, diff, threshold, scale[j]);
                return false;
            }
        }
    }
    return true;
}

void checkQuantParameters(const GRUQuantitativeParameters &quant_parms,
    const Tensor1f &bx,
    const Tensor1f &br,
    const Tensor2f &W,
    const Tensor2f &R,
    const Tensor3f &x,
    const Tensor3f &dh,
    const Tensor1i32 &bx_quant,
    const Tensor1i32 &br_quant,
    const Tensor2i8 &W_quant,
    const Tensor2i8 &R_quant,
    const Tensor3i8 &x_quant,
    const Tensor3i8 &dh_new_quant) {
    checkScale(x.data(), x.size(), x_quant.data(), quant_parms.scale_x_, quant_parms.zp_x_);
    checkScale(dh.data(), dh.size(), dh_new_quant.data(), quant_parms.scale_h_, quant_parms.zp_h_);
    checkScalePerChannel(W.data(), HIDDEN_DIMS * 3, INPUT_DIMS, W_quant.data(), quant_parms.scale_W_);
    checkScalePerChannel(R.data(), HIDDEN_DIMS * 3, HIDDEN_DIMS, R_quant.data(), quant_parms.scale_R_);
    checkScalePerChannel(bx.data(), HIDDEN_DIMS * 3, 1, bx_quant.data(), quant_parms.scale_bx_);
    checkScalePerChannel(br.data(), HIDDEN_DIMS * 3, 1, br_quant.data(), quant_parms.scale_br_);
}

int main() {
    srand(time(0));

    init_gru_cublas();  // 使用初始化函数

    // Weights.
    Tensor2f W(HIDDEN_DIMS * 3, INPUT_DIMS);  // 对应W_z/W_r/W_h的合并
    Tensor2f R(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
    Tensor1f bx(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 "输入 x_t 到门控的线性变换" 加偏置
    Tensor1f br(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给"隐藏状态 h_{t-1} 到门控的线性变换" 加偏置

    // Input.
    Tensor3f x(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);

    // Gradients from upstream layers.
    Tensor3f dh(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

    // W: 输入权重矩阵，使用 Xavier/Glorot 均匀初始化
    // 范围: U(-k, k)，其中 k = sqrt(6 / (input_size + hidden_size * 3))
    // 这确保前向和反向传播的方差保持稳定
    W.setRandom();
    float k_W = sqrtf(6.0f / (static_cast<float>(INPUT_DIMS) + static_cast<float>(HIDDEN_DIMS * 3)));
    W = W * W.constant(k_W);  // 将 [-1, 1] 缩放到 [-k_W, k_W]

    // R: 循环权重矩阵，使用较小的初始化范围
    // 范围: U(-k, k)，其中 k = sqrt(1 / hidden_size) 或更保守的 k = 1 / sqrt(hidden_size)
    // 循环权重通常需要更小的初始值以避免梯度爆炸
    R.setRandom();
    float k_R = 1.0f / sqrtf(static_cast<float>(HIDDEN_DIMS));
    R = R * R.constant(k_R);

    // bx, br: 偏置通常初始化为0或很小的随机值
    // PyTorch GRU 默认偏置为0，这里使用很小的随机值 [-0.01, 0.01] 以增加一些随机性
    bx.setRandom();
//    bx = bx * bx.constant(0.01f);  // 偏置缩放为 [-0.01, 0.01]
    br.setRandom();
//    br = br * br.constant(0.01f);  // 偏置缩放为 [-0.01, 0.01]

    // x: 输入数据，通常在 [-1, 1] 范围内（标准化后的输入）
    // 或者根据实际应用场景，可以是归一化后的数据
    // 这里使用 [-1, 1] 范围，模拟标准化后的输入
    x.setRandom();  // Eigen setRandom() 默认生成 [-1, 1] 的均匀分布

    // dh: 来自上层或损失函数的梯度，通常在 [-0.01, 0.01] 范围内
    // 实际训练中梯度值通常较小，这里使用合理的范围
    dh.setRandom();
    dh = dh * dh.constant(0.01f);  // 梯度缩放为 [-0.01, 0.01]


    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // 效验得到固定量化参数
    GRUQuantitativeParameters quant_parms;
    calibrateGruScales<int8_t>(W, R, bx, br, x, quant_parms);

    // Quant
    Tensor2i8 W_quant(HIDDEN_DIMS * 3, INPUT_DIMS);  // 对应W_z/W_r/W_h的合并
    Tensor2i8 R_quant(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
    Tensor1i32 bx_quant(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
    Tensor1i32 br_quant(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给“隐藏状态 h_{t-1} 到门控的线性变换” 加偏置
    Tensor3i8 x_quant(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);
    Tensor3i8 dh_new_quant(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

    // 使用固定量化参数将输入量化
    GruQuantInit<false>(W,
                        R,
                        bx,
                        br,
                        x,
                        dh,
                        W_quant,
                        R_quant,
                        bx_quant,
                        br_quant,
                        x_quant,
                        dh_new_quant,
                        quant_parms);

    checkQuantParameters(quant_parms, bx, br, W, R, x, dh, bx_quant, br_quant, W_quant, R_quant, x_quant, dh_new_quant);

    checkQuant(W_quant,
               R_quant,
               bx_quant,
               br_quant,
               x_quant,
               dh_new_quant);
    printGRUQuantitativeParameters(quant_parms);

    // 运行量化GRU得到量化结果2
    Tensor3i8 h_quant_inference(hidden_size, batch_size, (time_steps + 1));
    h_quant_inference.setZero();
    GruInferenceQuant(W_quant,
                      R_quant,
                      bx_quant,
                      br_quant,
                      x_quant,
                      quant_parms,
                      h_quant_inference);

    // 运行浮点GRU得到结果1
    Tensor3f h_inference(hidden_size, batch_size, (time_steps + 1));
    h_inference.setZero();
    GruInference(W, R, bx, br, x, h_inference);

    printf("cudaError(GruInference finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    if (true) { // Test
        std::vector<float> h_inference_tmp(h_inference.data(), h_inference.data() + h_inference.size());
        std::vector<int8_t> h_quant_inference_tmp(h_quant_inference.data(),
                                                  h_quant_inference.data() + h_quant_inference.size());

        checkHQuantizationWithCosine(h_inference_tmp,
                                     h_quant_inference_tmp,
                                     time_steps,
                                     batch_size,
                                     hidden_size,
                                     quant_parms);
    }

    printf("cudaError(GruInferenceQuant finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    GruTrain(W, R, bx, br, x, dh, false, false);

    printf("cudaError(GruTrain finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasDestroy(g_blas_handle);

    return 0;
}
