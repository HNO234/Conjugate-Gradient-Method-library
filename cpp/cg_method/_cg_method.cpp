#include <_cg_method.hpp>
#include <algorithm>
#include <cmath>

namespace cg_method{

Matrix::Naive_Matrix linear_CG::solve_by_Naive_Matrix(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x){
    /*
    Functionality: solve linear equation Ax = b by Conjugate Gradient method
    Parameter:
        A: Naive_Matrix, the coefficient matrix
        b: Naive_Matrix, the right hand side
        x: Naive_Matrix, the initial guess
    */
    Matrix::Naive_Matrix res;
    Matrix::Naive_Matrix delta;
    Matrix::Naive_Matrix D;
    res = A * x - b; //1d array
    delta = -res; //1d array
    for (int i = 0; i < epoch; ++i)
    {
        if (res.norm() < epsilon)
        {
            break;
        }
        D = A * delta; //1d array
        beta = -(res * delta)(0, 0) / (delta * D)(0, 0); //scalar
        x = x + (delta * beta); //1d array
        res = A * x - b; //1d array
        chi = (res * D)(0, 0) / (delta * D)(0, 0); //scalar
        delta = delta * chi -  res; //1d array
    }
    return x;
}

Matrix::Accelerated_Matrix linear_CG::solve_by_Accelerated_Matrix(Matrix::Accelerated_Matrix A, Matrix::Accelerated_Matrix b, Matrix::Accelerated_Matrix x){
    /*
    Functionality: solve linear equation Ax = b by Conjugate Gradient method
    Parameter:
        A: Accelerated_Matrix, the coefficient matrix
        b: Accelerated_Matrix, the right hand side
        x: Accelerated_Matrix, the initial guess
    */
    Matrix::Accelerated_Matrix res;
    Matrix::Accelerated_Matrix delta;
    Matrix::Accelerated_Matrix D;
    if(number_of_threads == 0){
        throw std::out_of_range("CG method Number of thread is 0.");
    } 
    A.set_number_of_threads() = number_of_threads;
    b.set_number_of_threads() = number_of_threads;
    x.set_number_of_threads() = number_of_threads;
    res.set_number_of_threads() = number_of_threads;
    delta.set_number_of_threads() = number_of_threads;
    D.set_number_of_threads() = number_of_threads;

    res = A * x - b; //1d array
    delta = -res; //1d array
    auto naive_dot_product = [](Matrix::Accelerated_Matrix &a_mat,
                                Matrix::Accelerated_Matrix &b_mat) {
        double ret = 0;
        auto total_size = a_mat.nrow() * a_mat.ncol();
        for (auto i = size_t{}; i < total_size; i++)
            ret += a_mat.buffer(i) * b_mat.buffer(i);
        return ret;
    };
    auto naive_iadd_with_scalar = [](Matrix::Accelerated_Matrix &a_mat,
                                    Matrix::Accelerated_Matrix &b_mat,
                                    double beta) {
        Matrix::Accelerated_Matrix ret = a_mat;
        auto total_size = a_mat.nrow() * a_mat.ncol();
        for (auto i = size_t{}; i < total_size; i++)
           a_mat(i, 0) += beta * b_mat.buffer(i);
        return ret;
    };
    for (int i = 0; i < epoch; ++i)
    {
        if (res.norm() <= epsilon)
            break;
        D = A * delta;
#pragma omp parallel num_threads(number_of_threads)
    {
        
#pragma omp single
      {
#pragma omp task shared(res, delta)
        res_dot_delta = naive_dot_product(res,delta);
#pragma omp task shared(delta, D)
        delta_dot_D = naive_dot_product(delta, D);
#pragma omp task shared(res, D)
        res_dot_D = naive_dot_product(res, D);
        // run on the first thread
        D_dot_D = naive_dot_product(D, D);
      }
    }
        beta = -res_dot_delta / delta_dot_D;
#pragma omp parallel num_threads(number_of_threads)
    {
        
#pragma omp single
      {
#pragma omp task shared(x, delta, beta)
        naive_iadd_with_scalar(x, delta, beta);
        // run on the first thread
        naive_iadd_with_scalar(res, D, beta);
      }
    }
        chi = (res_dot_D + beta * D_dot_D) / (delta_dot_D);
        delta *= chi;
        delta -= res;
    }
    return x;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Fletcher_Reeves_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta){
    /*
    Functionality: calculate the next iteration of delta by Fletcher Reeves method
    Parameter:
        cur_Df: Naive_Matrix, the gradient of the objective function at current iteration
        next_Df: Naive_Matrix, the gradient of the objective function at next iteration
        delta: Naive_Matrix, the direction of the next iteration
    */
    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Hager_Zhang_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta){
    /*
    Functionality: calculate the next iteration of delta by Hager Zhang method
    Parameter:
        cur_Df: Naive_Matrix, the gradient of the objective function at current iteration
        next_Df: Naive_Matrix, the gradient of the objective function at next iteration
        delta: Naive_Matrix, the direction of the next iteration
    */
    Matrix::Naive_Matrix Q = next_Df - cur_Df;
    Matrix::Naive_Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    Matrix::Naive_Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    double chi = (M * N)(0, 0);
    delta = - next_Df + delta * chi;
    return delta;    
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Dai_Yuan_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta){
    /* 
    Functionality: calculate the next iteration of delta by Dai Yuan method
    Parameter:
        cur_Df: Naive_Matrix, the gradient of the objective function at current iteration
        next_Df: Naive_Matrix, the gradient of the objective function at next iteration
        delta: Naive_Matrix, the direction of the next iteration
    */
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = - next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Fletcher_Reeves_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads){
    /*
    Functionality: calculate the next iteration of delta by Fletcher Reeves method
    Parameter:
        cur_Df: Accelerated_Matrix, the gradient of the objective function at current iteration
        next_Df: Accelerated_Matrix, the gradient of the objective function at next iteration
        delta: Accelerated_Matrix, the direction of the next iteration
    */
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    
    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Hager_Zhang_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads){
    /*
    Functionality: calculate the next iteration of delta by Hager Zhang method
    Parameter:
        cur_Df: Accelerated_Matrix, the gradient of the objective function at current iteration
        next_Df: Accelerated_Matrix, the gradient of the objective function at next iteration
        delta: Accelerated_Matrix, the direction of the next iteration
    */
    
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    
    Matrix::Accelerated_Matrix Q = next_Df - cur_Df;
    Matrix::Accelerated_Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    Matrix::Accelerated_Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    double chi = (M * N)(0, 0);
    delta = - next_Df + delta * chi;
    return delta;    
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Dai_Yuan_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads){
    /*
    Functionality: calculate the next iteration of delta by Dai Yuan method
    Parameter:
        cur_Df: Accelerated_Matrix, the gradient of the objective function at current iteration
        next_Df: Accelerated_Matrix, the gradient of the objective function at next iteration
        delta: Accelerated_Matrix, the direction of the next iteration
    */
    
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = - next_Df + delta * chi;
    return delta;
}

}