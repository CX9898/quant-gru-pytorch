//#include <ATen/cuda/CUDAContext.h>
//#include <c10/cuda/CUDAGuard.h>
//#include <torch/extension.h>
//#include <vector>
//#include <torch/types.h>
//#include <torch/python.h>
//
//#include "gru_quant.h"
//#include "support.h"
//
//using torch::Tensor;
//
//std::vector<Tensor> gru_forward(
//    bool training,
//    float zoneout_prob,
//    Tensor x,
//    Tensor h0,
//    Tensor kernel,
//    Tensor recurrent_kernel,
//    Tensor bias,
//    Tensor recurrent_bias,
//    Tensor zoneout_mask) {
//    const auto time_steps = x.size(0);
//    const auto batch_size = x.size(1);
//    const auto input_size = x.size(2);
//    const auto hidden_size = recurrent_kernel.size(0);
//    const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);
//
////    CHECK_INPUT(x);
////    CHECK_INPUT(h0);
////    CHECK_INPUT(kernel);
////    CHECK_INPUT(recurrent_kernel);
////    CHECK_INPUT(bias);
////    CHECK_INPUT(recurrent_bias);
////    CHECK_INPUT(zoneout_mask);
//
//    const auto options = x.options();
//    const at::cuda::CUDAGuard guard(options.device_index());
//    Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
//    Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, options);
//    Tensor tmp_Wx = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
//    Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 3 }, options);
//
//    output[0] = h0;
//
//    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gru_forward", ([&] {
//      gru::ForwardPassQuant<typename native_type<scalar_t>::T> forward(
//          training,
//          batch_size,
//          input_size,
//          hidden_size,
//          at::cuda::getCurrentCUDABlasHandle(),
//          at::cuda::getCurrentCUDAStream());
//
//      forward.Run(
//          time_steps,
//          ptr<scalar_t>(kernel),
//          ptr<scalar_t>(recurrent_kernel),
//          ptr<scalar_t>(bias),
//          ptr<scalar_t>(recurrent_bias),
//          ptr<scalar_t>(x),
//          ptr<scalar_t>(output),
//          ptr<scalar_t>(cache),
//          ptr<scalar_t>(tmp_Wx),
//          ptr<scalar_t>(tmp_Rh),
//          has_zoneout ? zoneout_prob : 0.0f,
//          has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr);
//    }));
//
//    return { output, cache };
//}
//
