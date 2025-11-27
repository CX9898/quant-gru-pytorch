// GRUQuantWrapper_pybind.cpp
#include <torch/extension.h>
#include "GRUQuantWrapper.hpp"

template<typename QuantT>
std::shared_ptr<GRUQuantWrapper<QuantT>> createWrapper(
    int T, int B, int input_size, int hidden_size) {
    return std::make_shared<GRUQuantWrapper<QuantT>>(T, B, input_size, hidden_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GRUQuantWrapper<int8_t>, std::shared_ptr<GRUQuantWrapper<int8_t>>>(m, "GRUQuantInt8")
        .def(py::init<int,int,int,int>())
        .def("initWeights", &GRUQuantWrapper<int8_t>::initWeights)
        .def("forward", &GRUQuantWrapper<int8_t>::forward)
        .def("backward", &GRUQuantWrapper<int8_t>::backward);

    py::class_<GRUQuantWrapper<int16_t>, std::shared_ptr<GRUQuantWrapper<int16_t>>>(m, "GRUQuantInt16")
        .def(py::init<int,int,int,int>())
        .def("initWeights", &GRUQuantWrapper<int16_t>::initWeights)
        .def("forward", &GRUQuantWrapper<int16_t>::forward)
        .def("backward", &GRUQuantWrapper<int16_t>::backward);
}