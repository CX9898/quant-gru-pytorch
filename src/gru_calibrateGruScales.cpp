#include <cublas_v2.h>

#include "gru.h"

void calibrateGruScales(bool use_int16,
                        int time_steps, int batch_size, int input_size, int hidden_size,
                        const float *W,
                        const float *R,
                        const float *bx,
                        const float *br,
                        const float *x,
                        const cublasHandle_t &g_blas_handle,
                        GRUQuantitativeParameters &quant_gru_scales
) {

    // Copy weights over to GPU.
    dev::vector<float> W_dev(W, hidden_size * 3 * input_size);
    dev::vector<float> R_dev(R, hidden_size * 3 * hidden_size);
    dev::vector<float> bx_dev(bx, hidden_size * 3);
    dev::vector<float> br_dev(br, hidden_size * 3);
    dev::vector<float> x_dev(x, input_size * batch_size * time_steps);
//    dev::vector<float> dh_new_dev(dh_new);

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        true,  // training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.setCalibrationMode(true, use_int16);

    forward.Run(
        time_steps,
        W_dev.data(),
        R_dev.data(),
        bx_dev.data(),
        br_dev.data(),
        x_dev.data(),
        h_dev.data(),
        v_dev.data(),
        tmp_Wx_dev.data(),
        tmp_Rh_dev.data(),
        0.0f,
        nullptr);

    quant_gru_scales = forward.getGRUQuantitativeParameters();
}