import sys
import os
# Get the absolute path of the directory containing the .so file
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../cpp"
# Add the directory to the Python path
sys.path.append(dir_path)

import math
import numpy as np
import random
from autograd import grad
import autograd.numpy as au
from _cgpy import CG
from _cgpy.Matrix import Naive_Matrix
from _cgpy.Matrix import Accelerated_Matrix

def is_pos_def(x): 
    """
    Functionality: Check if a matrix is positive definite
    Parameters:
    x: The input matrix.
    """
    return np.all(np.linalg.eigvals(x) > 0)

def is_symmetric(x): 
    """
    Functionality: Check if a matrix is symmetric
    Parameters:
    x: The input matrix.
    """
    return np.allclose(x, x.T)

def generate_pos_def_symmetric(n, cond, max_value):
    """
    Functionality: Generate a random positive definite symmetric matrix
    Parameters:
    n: The dimension of the matrix.
    cond: The fixed condition number.
    max_value: The maximum value in a matrix
    """
    def generate_pos_def(n):
        """
        Functionality: Generate a random positive definite matrix
        Parameters:
        n: The dimension of the matrix.
        """
        A = np.random.rand(n, n)
        return A.dot(A.T)

    def generate_symmetric(A):
        """
        Functionality: Generate a symmetric matrix from the input matrix.
        Parameters:
        A: The input matrix.
        """
        return (A + A.T)/2
    # Initialize a random positive definite symmetric matrix.
    a = generate_pos_def(n)
    a = generate_symmetric(a)
    # Adjust the singlar values to fix the condition number.
    # Since cond = s[0] / s[n - 1], by linear scaling s, we can fix the condition number.
    u, s, v = np.linalg.svd(a)
    news = s[0] * (np.full((n), 1) - ((cond - 1) / cond) * (np.full((n), s[0]) - s) / (s[0] - s[n - 1]))
    news_mat = np.diag(news)
    a = np.dot(u, np.dot(news_mat, v))
    # Adjust the maximum value in a
    current_max_value = a.max()
    a = a / current_max_value * max_value
    assert(is_pos_def(a) and is_symmetric(a))
    return a

def custom_linear_CG(x, a, b, epsilon = 5e-7, epoch=10000000, use_accelerated = True, num_threads = 3):
    """
    Functionality: Solve the linear system Ax = b using conjugate gradient method by calling the C++ library
    Parameters:
    x: The initial point.
    a: The matrix A.
    b: The vector b.
    epsilon: The tolerance.
    epoch: The maximum number of iterations.
    num_threads: The number of threads used in the C++ library.
    """
    mat_x_min = None
    if(num_threads <= 0):
        num_threads = 1
    if(use_accelerated):
        mat_a = Accelerated_Matrix(a)
        mat_b = Accelerated_Matrix(b)
        mat_x = Accelerated_Matrix(x)
        mat_x.set_number_of_threads(num_threads)
        #mat_a.set_number_of_threads(num_threads)
        #mat_b.set_number_of_threads(num_threads)
        linear_cg_model = CG.linear_CG(epsilon, epoch, num_threads)
        mat_x_min = linear_cg_model.solve_by_Accelerated_Matrix(mat_a, mat_b, mat_x)


    else:
        mat_a = Naive_Matrix(a)
        mat_b = Naive_Matrix(b)
        mat_x = Naive_Matrix(x)
        linear_cg_model = CG.linear_CG(epsilon, epoch)
        mat_x_min = linear_cg_model.solve_by_Naive_Matrix(mat_a, mat_b, mat_x)
    return np.array(mat_x_min.tolist())

def np_linear_CG(x, A, b, epsilon, epoch=10000000):
    """
    Functionality: Solve the linear system Ax = b using conjugate gradient method by calling the numpy library
    Parameters:
    x: The initial point.
    A: The matrix A.
    b: The vector b.
    epsilon: The tolerance.
    epoch: The maximum number of iterations.
    """
    res = A.dot(x) - b
    delta = -res 
    count = 0    
    while True:
        
        if (np.linalg.norm(res) <= epsilon) or (count >= epoch):
            return x, count
        
        D = A.dot(delta)
        delta_dot_D = (delta.dot(D)) 
        beta = -(res.dot(delta))/delta_dot_D
        x = x + beta*delta

        res = A.dot(x) - b 
        chi = res.dot(D)/delta_dot_D
        delta = chi*delta -  res 

        count += 1
        #print("A", A)
        #print("B", b)
        #print("x", x)
        #print("res", res)
        #print("delta", delta)
        #print("D", D)
        #print("beta", beta)
        #print("chi", chi)
        #print("norm", np.linalg.norm(res))
        #print("------------------")

def nonlinear_func_1(x):
    """
    Functionality: f(x) = x1^4 - 2*x1^2*x2 + x1^2 + x2^2 - 2*x1 + 1
    Parameters: 
    x: The input vector.
    """
    return x[0]**4 - 2*x[0]**2*x[1] + x[0]**2 + x[1]**2 - 2*x[0] + 1

def nonlinear_func_2(x):
    """
    Functionality: f(x) = x1^4 + x2^4 + x3^4 + x4^4 ...+ xn^4
    Parameters: 
    x: The input vector.
    """
    x = np.array(x)
    return (np.sum(x**4))**0.5

def custom_naive_line_search(f, df, x, d, alpha=5e-4, beta=0.8):
    """
    Functionality: Perform a backtracking line search to find the step size.
    Parameters:
    f: The objective function.
    df: The gradient of the objective function.
    x: The current point.
    d: The search direction.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    """
    Mat_df_x = Naive_Matrix(df(x)) #df(x)
    Mat_d = Naive_Matrix(d) #d
    Mat_df_x_dot_d_mul_alpha = (Mat_df_x @ Mat_d)[0, 0] * alpha #df(x).dot(d) * alpha
    f_x = f(x) #f(x)

    t = 1.0
    Mat_x_p_t_d = Naive_Matrix(x) + Mat_d * t #x + t * d

    while f(Mat_x_p_t_d.tolist()) > (Mat_df_x_dot_d_mul_alpha * t + f_x):
        t *= beta
        Mat_x_p_t_d = Naive_Matrix(x) + Mat_d * t
    return t

def custom_accelerated_line_search(f, df, x, d, alpha=5e-4, beta=0.8, num_threads = 1):
    """
    Functionality: Perform a backtracking line search to find the step size.
    Parameters:
    f: The objective function.
    df: The gradient of the objective function.
    x: The current point.
    d: The search direction.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    """
    ##print("num_threads", num_threads)
    Mat_df_x = Accelerated_Matrix(df(x)) #df(x)
    Mat_d = Accelerated_Matrix(d) #d
    Mat_df_x.set_number_of_threads(num_threads)
    #Mat_d.set_number_of_threads(num_threads)
    Mat_df_x_dot_d_mul_alpha = (Mat_df_x @ Mat_d)[0, 0] * alpha #df(x).dot(d) * alpha
    f_x = f(x) #f(x)

    t = 1.0
    Mat_x_p_t_d = Accelerated_Matrix(x) + Mat_d * t #x + t * d
    #Mat_x_p_t_d.set_number_of_threads(num_threads)

    while f(Mat_x_p_t_d.tolist()) > (Mat_df_x_dot_d_mul_alpha * t + f_x):
        t *= beta
        Mat_x_p_t_d = Accelerated_Matrix(x) + Mat_d * t
    return t

def np_line_search(f, df, x, d, alpha=5e-4, beta=0.8):
    """
    Functionality: Perform a backtracking line search to find the step size.
    Parameters:
    f: The objective function.
    df: The gradient of the objective function.
    x: The current point.
    d: The search direction.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    """
    t = 1.0
    while f(x + t * d) > f(x) + alpha * t * np.dot(df(x), d):
        t *= beta
    return t



def custom_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves", use_accelerated = True, num_threads = -1):
    """
    Functionality: Solve the nonlinear system using conjugate gradient method by calling the c++ library
    Parameters:
    X: The initial point.
    tol: The tolerance.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    f: The objective function.
    Df: The gradient of the objective function.
    method: The method used to update the search direction.
    num_threads: The number of threads used in the C++ library.
    """
    if(not use_accelerated):
        return custom_naive_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves")
    
    else:
        return custom_accelerated_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves", num_threads = num_threads)
                
def custom_naive_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves"):
    """
    Functionality: Solve the nonlinear system using conjugate gradient method and naive matrix by calling the c++ library
    Parameters:
    X: The initial point.
    tol: The tolerance.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    f: The objective function.
    Df: The gradient of the objective function.
    method: The method used to update the search direction.
    """
    method_dict = {}
    method_dict = {
                "Fletcher_Reeves": CG.nonlinear_CG.Naive_Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": CG.nonlinear_CG.Naive_Hager_Zhang_next_iteration,\
                "Dai-Yuan": CG.nonlinear_CG.Naive_Dai_Yuan_next_iteration,\
    }

    if method in method_dict: 
        method_func = method_dict[method]
    else:
        raise AssertionError("method not supported")
    
    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 

    while True:
        start_point = X
        step = custom_naive_line_search(f = f, df = Df, x = start_point, d = delta, alpha=alpha, beta=beta)
        if step!=None:
            next_X = X+ step*delta
        elif step != step: #IsNaN
            raise AssertionError("It diverges, please try another start point or another hyperparameter.")
        else:
            return X, f(X)
        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            Mat_cur_Df = Naive_Matrix(cur_Df)
            Mat_next_Df = Naive_Matrix(next_Df)
            Mat_delta = Naive_Matrix(delta)
            Mat_delta = method_func(Mat_cur_Df, Mat_next_Df, Mat_delta)
            delta = np.array(Mat_delta.tolist())

def custom_accelerated_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves", num_threads = -1):
    """
    Functionality: Solve the nonlinear system using conjugate gradient method and accelerated matrix by calling the c++ library
    Parameters:
    X: The initial point.
    tol: The tolerance.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    f: The objective function.
    Df: The gradient of the objective function.
    method: The method used to update the search direction.
    num_threads: The number of threads used in the C++ library.
    """
    method_dict = {}
    method_dict = {
                "Fletcher_Reeves": CG.nonlinear_CG.Accelerated_Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": CG.nonlinear_CG.Accelerated_Hager_Zhang_next_iteration,\
                "Dai-Yuan": CG.nonlinear_CG.Accelerated_Dai_Yuan_next_iteration,\
    }

    if method in method_dict: 
        method_func = method_dict[method]
    else:
        raise AssertionError("method not supported")
    
    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 

    while True:
        start_point = X
        step = custom_accelerated_line_search(f = f, df = Df, x = start_point, d = delta, alpha=alpha, beta=beta, num_threads = num_threads)
        if step!=None:
            next_X = X+ step*delta 
        elif step != step: #IsNaN
            raise AssertionError("It diverges, please try another start point or another hyperparameter.")
        else:
            return X, f(X)

        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            Mat_cur_Df = Accelerated_Matrix(cur_Df)
            Mat_next_Df = Accelerated_Matrix(next_Df)
            Mat_delta = Accelerated_Matrix(delta)
            Mat_delta = method_func(Mat_cur_Df, Mat_next_Df, Mat_delta, num_threads)
            delta = np.array(Mat_delta.tolist())

def np_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves"):
    """
    Functionality: Solve the nonlinear system using conjugate gradient method by calling the numpy library
    Parameters:
    X: The initial point.
    tol: The tolerance.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    f: The objective function.
    Df: The gradient of the objective function.
    method: The method used to update the search direction.
    """
    method_dict = {
                "Fletcher_Reeves": Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": Hager_Zhang_next_iteration,\
                "Dai-Yuan": Dai_Yuan_next_iteration,\
    }

    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 

    if method in method_dict: 
        method_func = method_dict[method]
    else:
        raise AssertionError("method not supported")

    while True:
        start_point = X
        step = np_line_search(f = f, df = Df, x = start_point, d = delta, alpha=alpha, beta=beta)
        if step!=None:
            next_X = X+ step*delta 
        elif step != step: #IsNaN
            raise AssertionError("It diverges, please try another start point or another hyperparameter.")
        else:
            return X, f(X)

        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            delta = method_func(cur_Df, next_Df, delta)
            delta = np.array(delta.tolist())
            
def Fletcher_Reeves_next_iteration(cur_Df, next_Df, delta):
    """
    Functionality: Update the search direction using Fletcher Reeves method
    Parameters:
    cur_Df: The gradient at the current point.
    next_Df: The gradient at the next point.
    delta: The search direction.
    """
    chi = np.linalg.norm(next_Df)**2/np.linalg.norm(cur_Df)**2
    delta = -next_Df + chi*delta
    return delta

def Hager_Zhang_next_iteration(cur_Df, next_Df, delta):
    """
    Functionality: Update the search direction using Hager Zhang method
    Parameters:
    cur_Df: The gradient at the current point.
    next_Df: The gradient at the next point.
    delta: The search direction.
    """
    Q = next_Df - cur_Df
    M = Q - delta * (np.linalg.norm(Q)**2) * 2/(delta.dot(Q))
    N = next_Df/(delta.dot(Q))
    chi = M.dot(N)
    delta = -next_Df + chi*delta
    return delta

def Dai_Yuan_next_iteration(cur_Df, next_Df, delta):
    """
    Functionality: Update the search direction using Dai Yuan method
    Parameters:
    cur_Df: The gradient at the current point.
    next_Df: The gradient at the next point.
    delta: The search direction.
    """
    chi = np.linalg.norm(next_Df)**2/delta.dot(next_Df - cur_Df)
    delta = -next_Df + chi*delta
    return delta