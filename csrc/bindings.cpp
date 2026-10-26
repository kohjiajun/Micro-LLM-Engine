#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "engine.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;

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

py::array_t<float> py_fp32_gemm(FloatArray A, FloatArray B){
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    if(buf_A.ndim != 2 || buf_B.ndim != 2){
        throw std::runtime_error("Input Arrays must be 2D");
    }
    const size_t M = static_cast<size_t>(buf_A.shape[0]);
    const size_t KA = static_cast<size_t>(buf_A.shape[1]);
    const size_t KB = static_cast<size_t>(buf_B.shape[0]);
    const size_t N = static_cast<size_t>(buf_B.shape[1]);

    if(KA != KB){
        throw std::runtime_error("COL A must be equal to ROW B");
    }

    py::array_t<float> result ({
        static_cast<py::ssize_t>(M),
        static_cast<py::ssize_t>(N)
    });
    float *res_ptr = result.mutable_data();
    {
        py::gil_scoped_release release;
        fp32_gemm(
            static_cast<const float*>(buf_A.ptr),
            static_cast<const float*>(buf_B.ptr),
            res_ptr,
            M,
            N,
            KA
        );
    }
    return result;
}

py::array_t<int32_t> py_int8_gemm(IntArray A, IntArray B){
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();
    
    if(buf_A.ndim != 2 || buf_B.ndim != 2){
        throw std::runtime_error("Input Arrays must be 2D");
    }
    const size_t M = static_cast<size_t>(buf_A.shape[0]);
    const size_t KA = static_cast<size_t>(buf_A.shape[1]);
    const size_t KB = static_cast<size_t>(buf_B.shape[0]);
    const size_t N = static_cast<size_t>(buf_B.shape[1]);

    if(KA != KB){
        throw std::runtime_error("COL A must be equal to ROW B");
    }
    

    py::array_t<int32_t> result({
        static_cast<py::ssize_t>(buf_A.shape[0]),
        static_cast<py::ssize_t>(buf_B.shape[1])
    });
    int32_t *res_ptr = result.mutable_data();
    {
        py::gil_scoped_release release;
        int8_gemm(
            static_cast<const int8_t*>(buf_A.ptr),
            static_cast<const int8_t*>(buf_B.ptr),
            res_ptr,
            M,N,KA
        );
    }
    return result;
}

PYBIND11_MODULE(micro_llm_engine,m){
    m.doc() = "High-performance C++ math kernels for dot product and quantization";

    m.def("fp32_dot_product", &py_fp32_dot_product, "Computes float32 dot product of two array");
    m.def("fp32_to_int8_quantization",&py_fp32_to_int8_quantization,"Quantize float23 array to int8 array");
    m.def("int8_dot_product",&py_int8_dot_product,"Computes int8 dot product of two array, returns int32 sum");
    m.def("fp32_gemm", &py_fp32_gemm,"Computes product of two fp32 matrixes");
    m.def("int8_gemm", &py_int8_gemm,"Computes product of two int8 matrixes");
}