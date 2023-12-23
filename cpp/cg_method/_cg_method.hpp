#ifndef __CG_METHOD_HPP__
#define __CG_METHOD_HPP__

#include <vector>
#include <_matrix_naive.hpp>
#include <_matrix_acc.hpp>
#include <algorithm>
#include <cmath>
#include <omp.h>
namespace cg_method{

class linear_CG{
    public:
        linear_CG() = default;
        linear_CG(double epsilon, int epoch, int number_of_threads): 
                epsilon(epsilon), epoch(epoch), number_of_threads(number_of_threads){};
        Matrix::Naive_Matrix solve_by_Naive_Matrix(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x);
        Matrix::Accelerated_Matrix solve_by_Accelerated_Matrix(Matrix::Accelerated_Matrix A, Matrix::Accelerated_Matrix b, Matrix::Accelerated_Matrix x);        
        double const & get_epsilon() const{ return epsilon; }
        double       & set_epsilon(){ return epsilon; }
        int const & get_epoch() const{ return epoch; }
        int       & set_epoch(){ return epoch; }
        int const & get_number_of_threads() const{ return number_of_threads; }
        int       & set_number_of_threads(){ return number_of_threads; }
    private:
        double epsilon = 5e-7;
        double beta = 0;
        double chi = 0;
        double res_dot_delta = 0;
        double delta_dot_D = 0;
        double res_dot_D = 0;
        double D_dot_D = 0;
        int epoch = 10000000;
        int number_of_threads = std::thread::hardware_concurrency();
};

class nonlinear_CG{
    public:
        static Matrix::Naive_Matrix Naive_Fletcher_Reeves_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
        static Matrix::Naive_Matrix Naive_Hager_Zhang_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
        static Matrix::Naive_Matrix Naive_Dai_Yuan_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
        static Matrix::Accelerated_Matrix Accelerated_Fletcher_Reeves_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads);
        static Matrix::Accelerated_Matrix Accelerated_Hager_Zhang_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads);
        static Matrix::Accelerated_Matrix Accelerated_Dai_Yuan_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads);        
};

}
#endif