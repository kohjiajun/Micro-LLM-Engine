#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "engine.h"

namespace py = pybind11;

float py_fp32_dot_product(py::array_t<float> A, py::array_t<float> B){
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    if(buf_A.size != buf_B.size){
        throw std::runtime_error("Input arrays must have matching lengths");
    }

    float result = 0.00f;

    {
        py::gil_scoped_release release;

        result = fp32_dot_product(
        static_cast<const float*>(buf_A.ptr),
        static_cast<const float*>(buf_B.ptr),
        buf_A.size);
    }
    return result;

}

py::tuple py_fp32_to_int8_quantization(py::array_t<float> input){
    py::buffer_info buf_input = input.request();

    size_t size = buf_input.size;

    auto q_out_array = py::array_t<int8_t>(size);
    py::buffer_info buf_q_out = q_out_array.request();
    float scale = 0.00f;

    {
        py::gil_scoped_release release;

        scale = fp32_to_int8_quantization(
        static_cast<const float*>(buf_input.ptr),
        static_cast<int8_t*>(buf_q_out.ptr),
        size);

    }

    return py::make_tuple(scale,q_out_array);
}

int32_t py_int8_dot_product(py::array_t<int8_t> A, py::array_t<int8_t> B){
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    if(buf_A.size != buf_B.size){
        throw std::runtime_error("Input arrays must have matching lengths");
    }

    int32_t result = 0;
    {
        py::gil_scoped_release release;
        result = int8_dot_product(
        static_cast<const int8_t*>(buf_A.ptr),
        static_cast<const int8_t*>(buf_B.ptr),
        buf_A.size);
    }
    return result;
}

PYBIND11_MODULE(micro_llm_engine,m){
    m.doc() = "High-performance C++ math kernels for dot product and quantization";

    m.def("fp32_dot_product", &py_fp32_dot_product, "Computes float32 dot product of two array");
    m.def("fp32_to_int8_quantization",&py_fp32_to_int8_quantization,"Quantize float23 array to int8 array");
    m.def("int8_dot_product",&py_int8_dot_product,"Computes int8 dot product of two array, returns int32 sum");
}