#include "../.venv/lib/python3.10/site-packages/pybind11/include/pybind11/pybind11.h"
#include "../.venv/lib/python3.10/site-packages/pybind11/include/pybind11/numpy.h"
#include "../eigen/Eigen/Dense"

namespace py = pybind11;

// Function that accepts numpy array and returns a numpy array after some Eigen operations
py::array_t<double> multiply_matrix_array(const py::array_t<double>& input) {
    // Convert numpy array to Eigen matrix
    auto buf = input.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input should be a 2D numpy array");
    }

    Eigen::Map<const Eigen::MatrixXd> mat(static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]);

    // Perform some operation on the matrix
    Eigen::MatrixXd result = mat * mat.transpose();

    // Convert Eigen matrix back to numpy array
    py::array_t<double> output({result.rows(), result.cols()});
    auto buf_out = output.request();
    double* ptr_out = static_cast<double*>(buf_out.ptr);

    for (ssize_t i = 0; i < result.rows(); i++) {
        for (ssize_t j = 0; j < result.cols(); j++) {
            ptr_out[i * result.cols() + j] = result(i, j);
        }
    }

    return output;
}

py::array_t<double> multiply_matrix(const py::array_t<double>& input) {
    // Convert numpy array to Eigen matrix
    auto buf = input.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input should be a 2D numpy array");
    }

    Eigen::Map<const Eigen::MatrixXd> mat(static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]);

    // Perform some operation on the matrix
    Eigen::MatrixXd result = mat * mat.transpose();

    // Create a numpy array with the same dimensions as the result
    py::array_t<double> output({result.rows(), result.cols()});

    // Map the numpy array to an Eigen matrix
    Eigen::Map<Eigen::MatrixXd>(static_cast<double*>(output.mutable_data()), result.rows(), result.cols()) = result;

    return output;
}

// Bindings to Python
PYBIND11_MODULE(EigenForPython, m) {
    m.def("multiply_matrix_array", &multiply_matrix_array, "Multiply matrix with its transpose and return array");
    m.def("multiply_matrix", &multiply_matrix, "Multiply matrix with its transpose and return eigen object", py::return_value_policy::reference);
}
