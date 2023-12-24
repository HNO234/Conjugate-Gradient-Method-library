#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <utility>
#include <_matrix_naive.hpp>
#include <_matrix_acc.hpp>
#include <_cg_method.hpp>
#include <omp.h>
namespace py = pybind11;

PYBIND11_MODULE(_cgpy, m){
    auto m_matrix = m.def_submodule("Matrix");
    py::class_<Matrix::Naive_Matrix>(m_matrix,"Naive_Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, std::vector<double> const &>())
        .def(py::init<std::vector<double> const &>())
        .def(py::init<Matrix::Naive_Matrix const&>())
        .def(py::init<std::vector<std::vector<double>> const &>())
        .def("__getitem__", [](Matrix::Naive_Matrix &mat, std::pair<size_t, size_t> index) -> double{
	        return mat(index.first, index.second);
	    })
        .def("__repr__", [](Matrix::Naive_Matrix &mat) -> std::vector<double>{
	        return mat.buffer_vector();
	    })
	    .def("__setitem__", [](Matrix::Naive_Matrix &mat, std::pair<size_t, size_t> index, double val){
		    mat(index.first, index.second) = val;
	    })
        .def("__eq__", &Matrix::Naive_Matrix::operator==)
        .def("__add__", &Matrix::Naive_Matrix::operator+)
        .def("__add__", &Matrix::Naive_Matrix::operator+)
        .def("__iadd__", &Matrix::Naive_Matrix::operator+=)
        .def("__sub__", static_cast<Matrix::Naive_Matrix (Matrix::Naive_Matrix::*)(Matrix::Naive_Matrix const &) const>(&Matrix::Naive_Matrix::operator-))
        .def("__isub__", &Matrix::Naive_Matrix::operator-=)
        .def("__neg__", static_cast<Matrix::Naive_Matrix (Matrix::Naive_Matrix::*)() const>(&Matrix::Naive_Matrix::operator-))
        .def("__mul__", static_cast<Matrix::Naive_Matrix (Matrix::Naive_Matrix::*)(double const &) const>(&Matrix::Naive_Matrix::operator*))
        .def("__imul__", &Matrix::Naive_Matrix::operator*=)
        .def("__matmul__", static_cast<Matrix::Naive_Matrix (Matrix::Naive_Matrix::*)(Matrix::Naive_Matrix const &) const>(&Matrix::Naive_Matrix::operator*))
        .def("norm", &Matrix::Naive_Matrix::norm)
        .def_property_readonly("nrow", &Matrix::Naive_Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::Naive_Matrix::ncol)
        .def("tolist", &Matrix::Naive_Matrix::buffer_vector)
        .def("tolist2d", &Matrix::Naive_Matrix::buffer_vector2d)
        ;
    py::class_<Matrix::Accelerated_Matrix>(m_matrix,"Accelerated_Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, std::vector<double> const &>())
        .def(py::init<std::vector<double> const &>())
        .def(py::init<Matrix::Accelerated_Matrix const&>())
        .def(py::init<std::vector<std::vector<double>> const &>())
        .def("__getitem__", [](Matrix::Accelerated_Matrix &mat, std::pair<size_t, size_t> index) -> double{
	        return mat(index.first, index.second);
	    })
        .def("__repr__", [](Matrix::Accelerated_Matrix &mat) -> std::vector<double>{
	        return mat.buffer_vector();
	    })
	    .def("__setitem__", [](Matrix::Accelerated_Matrix &mat, std::pair<size_t, size_t> index, double val){
		    mat(index.first, index.second) = val;
	    })
        .def("__eq__", &Matrix::Accelerated_Matrix::operator==)
        .def("__add__", &Matrix::Accelerated_Matrix::operator+)
        .def("__iadd__", &Matrix::Accelerated_Matrix::operator+=)
        .def("__sub__", static_cast<Matrix::Accelerated_Matrix (Matrix::Accelerated_Matrix::*)(Matrix::Accelerated_Matrix const &) const>(&Matrix::Accelerated_Matrix::operator-))
        .def("__isub__", &Matrix::Accelerated_Matrix::operator-=)
        .def("__neg__", static_cast<Matrix::Accelerated_Matrix (Matrix::Accelerated_Matrix::*)() const>(&Matrix::Accelerated_Matrix::operator-))
        .def("__mul__", static_cast<Matrix::Accelerated_Matrix (Matrix::Accelerated_Matrix::*)(double const &) const>(&Matrix::Accelerated_Matrix::operator*))
        .def("__imul__", &Matrix::Accelerated_Matrix::operator*=)
        .def("__matmul__", static_cast<Matrix::Accelerated_Matrix (Matrix::Accelerated_Matrix::*)(Matrix::Accelerated_Matrix const &) const>(&Matrix::Accelerated_Matrix::operator*))
        .def("set_number_of_threads", [](Matrix::Accelerated_Matrix &mat, int number_of_threads){
            mat.set_number_of_threads() = number_of_threads;
        })
        .def("get_number_of_threads", &Matrix::Accelerated_Matrix::get_number_of_threads)
        .def("norm", &Matrix::Accelerated_Matrix::norm)
        .def_property_readonly("nrow", &Matrix::Accelerated_Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::Accelerated_Matrix::ncol)
        .def("tolist", &Matrix::Accelerated_Matrix::buffer_vector)
        .def("tolist2d", &Matrix::Accelerated_Matrix::buffer_vector2d)
        ;    

    auto m_cg_method = m.def_submodule("CG");    
    py::class_<cg_method::linear_CG>(m_cg_method, "linear_CG")
        .def(py::init<double , int, int>(), py::arg("epsilon") = 5e-7, py::arg("epoch") = 10000000, py::arg("number_of_threads") = 1)
        .def("solve_by_Naive_Matrix", &cg_method::linear_CG::solve_by_Naive_Matrix)
        .def("solve_by_Accelerated_Matrix", &cg_method::linear_CG::solve_by_Accelerated_Matrix)
        .def("set_epsilon", &cg_method::linear_CG::set_epsilon)
        .def("get_epsilon", &cg_method::linear_CG::get_epsilon)
        .def("set_epoch", &cg_method::linear_CG::set_epoch)
        .def("get_epoch", &cg_method::linear_CG::get_epoch)
        .def("set_number_of_threads", &cg_method::linear_CG::set_number_of_threads)
        .def("get_number_of_threads", &cg_method::linear_CG::get_number_of_threads)
        ;
    py::class_<cg_method::nonlinear_CG>(m_cg_method, "nonlinear_CG")
        .def_static("Naive_Fletcher_Reeves_next_iteration", &cg_method::nonlinear_CG::Naive_Fletcher_Reeves_next_iteration)
        .def_static("Naive_Hager_Zhang_next_iteration", &cg_method::nonlinear_CG::Naive_Hager_Zhang_next_iteration)
        .def_static("Naive_Dai_Yuan_next_iteration", &cg_method::nonlinear_CG::Naive_Dai_Yuan_next_iteration)
        .def_static("Accelerated_Fletcher_Reeves_next_iteration", &cg_method::nonlinear_CG::Accelerated_Fletcher_Reeves_next_iteration)
        .def_static("Accelerated_Hager_Zhang_next_iteration", &cg_method::nonlinear_CG::Accelerated_Hager_Zhang_next_iteration)
        .def_static("Accelerated_Dai_Yuan_next_iteration", &cg_method::nonlinear_CG::Accelerated_Dai_Yuan_next_iteration)        
        ;
    //m.def("multiply_naive", &multiply_naive);
    //m.def("multiply_tile", &multiply_tile);
}