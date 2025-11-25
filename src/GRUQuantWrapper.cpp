#include "GRUQuantWrapper.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

PYBIND11_MODULE(gru_quant, m) {
    py::class_<GRUQuantWrapper>(m, "GRUQuantWrapper")
        .def(py::init<bool,int,int,int,int>(),
             py::arg("use_int16"),
             py::arg("time_steps"),
             py::arg("batch_size"),
             py::arg("input_size"),
             py::arg("hidden_size"))
        .def("initWeights", &GRUQuantWrapper::initWeights)
        .def("forward", &GRUQuantWrapper::forward);
}
