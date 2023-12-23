import sys
import os
# Get the absolute path of the directory containing the .so file
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../cpp"
# Add the directory to the Python path
sys.path.append(dir_path)
# Get the absolute path of the directory containing the utils.py
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../python"
# Add the directory to the Python path
sys.path.append(dir_path)

import pytest
import math
import time
import numpy as np
import utils
from autograd import grad
import autograd.numpy as au

from _cgpy import CG
from _cgpy.Matrix import Naive_Matrix
from _cgpy.Matrix import Accelerated_Matrix

def compare_linear_cg(n, cond, max_value, epoch, num_of_threads = 16):
    print("Compare linear CG")
    np.random.seed(0)
    a = utils.generate_pos_def_symmetric(n, cond, max_value)
    b = np.random.rand(n)
    x = np.random.rand(n)

    sum = 0.0
    for i in range(epoch):
        total = 0.0
        start = time.time()
        np_x = utils.np_linear_CG(x, a, b, 5e-7)
        end = time.time()
        total += end - start
        sum += total
        #print("Numpy for epoch ", i, " takes ", total, " seconds")
    np_total_avg = sum / epoch
    print("Numpy average time: ", np_total_avg)
    
    sum = 0.0
    for i in range(epoch):
        total = 0.0
        start = time.time()
        np_naive_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, epoch=10000, num_threads = 1)
        end = time.time()
        total += end - start
        sum += total
        #print("Naive for epoch ", i, " takes ", total, " seconds")
    np_total_avg = sum / epoch
    print("Naive average time: ", np_total_avg)

    for i in range(2, num_of_threads + 1):
        sum = 0.0
        for j in range(epoch):
            total = 0.0
            start = time.time()
            np_acc_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, epoch=10000, num_threads = i)
            end = time.time()
            total += end - start
            sum += total
            #print("Accelerated for epoch ", j, "and ", i," threads takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Accelerated average time for ", i, " threads: ", np_total_avg)

def compare_nonlinear_cg_func1(epoch,  num_of_threads = 16):
    np.random.seed(3)
    x_rand = np.random.uniform(low=3, high=5, size=(2,))
    print("Compare nonlinear CG")
    print("case 1")
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_x, _ = utils.np_nonlinear_CG(x, 5e-8, 0.5, 0.8, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method)
            end = time.time()
            total += end - start
            sum += total
            #print("Numpy for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Numpy average time for ", method, "in case 1 : ", np_total_avg)

        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_naive_mat_x_min, _ = utils.custom_nonlinear_CG(x, 5e-8, 0.5, 0.8, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method, 1)
            end = time.time()
            total += end - start
            sum += total
            #print("Custom for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Custom average time for ", method, "in case 1 : ", np_total_avg)


        for j in range(2, num_of_threads + 1):
            sum = 0.0
            x = np.copy(x_rand)
            for i in range(epoch):
                total = 0.0
                start = time.time()
                np_acc_mat_x_min, _ = utils.custom_nonlinear_CG(x, 5e-8, 0.5, 0.8, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method, j)
                end = time.time()
                total += end - start
                sum += total
                #print("Custom for epoch ", i, " takes ", total, " seconds")
            np_total_avg = sum / epoch
            print("Custom average time for ", method, " and ", j, " threads in case 1: ", np_total_avg)
     
def compare_nonlinear_cg_func2(epoch,  num_of_threads = 16):
    np.random.seed(3)
    x_rand = np.random.uniform(low=0.5, high=0.7, size=(100,))
    print("case 2")
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_x, _ = utils.np_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method)
            end = time.time()
            total += end - start
            sum += total
            #print("Numpy for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Numpy average time for ", method, "in case 1 : ", np_total_avg)

        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_acc_mat_x_min, _ = utils.custom_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method, 1)
            end = time.time()
            total += end - start
            sum += total
            #print("Custom for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Custom average time for ", method, "in case 1 : ", np_total_avg)


        for j in range(2, num_of_threads + 1):
            sum = 0.0
            x = np.copy(x_rand)
            for i in range(epoch):
                total = 0.0
                start = time.time()
                np_acc_mat_x_min, _ = utils.custom_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method, j)
                end = time.time()
                total += end - start
                sum += total
                #print("Custom for epoch ", i, " takes ", total, " seconds")
            np_total_avg = sum / epoch
            print("Custom average time for ", method, " and ", j, " threads in case 1: ", np_total_avg)

def compare_line_search_func2(epoch,  num_of_threads = 16, msize = 1000):
    np.random.seed(3)
    x_rand = np.random.uniform(low=0.5, high=0.7, size=(msize,))
    df = utils.grad(utils.nonlinear_func_2)
    d = - df(x_rand)
    print("compare line search for func2")
    sum = 0.0
    x_n = Naive_Matrix(x_rand)
    d_n = Naive_Matrix(d)
    for i in range(epoch):
        total = 0.0
        start = time.time()
        _ = utils.np_line_search(f = utils.nonlinear_func_2, df =df , x = x_rand , d = d)
        end = time.time()
        total += end - start
        sum += total
        #print("Numpy for epoch ", i, " takes ", total, " seconds")
    np_total_avg = sum / epoch
    print("Numpy average time for line search : ", np_total_avg)
    sum = 0.0
    x_n = Accelerated_Matrix(x_rand)
    d_n = Accelerated_Matrix(d)
    for i in range(epoch):
        total = 0.0
        start = time.time()
        _ = utils.custom_naive_line_search(f = utils.nonlinear_func_2, df =df , x = x_rand , d = d)
        end = time.time()
        total += end - start
        sum += total
        #print("Custom for epoch ", i, " takes ", total, " seconds")
    np_total_avg = sum / epoch
    print("Custom average time for line search : ", np_total_avg)
    for j in range(2, num_of_threads + 1):
        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            _ = utils.custom_accelerated_line_search(f = utils.nonlinear_func_2, df =df , x = x_rand, d = d, num_threads = j)
            end = time.time()
            total += end - start
            sum += total
        np_total_avg = sum / epoch
        print("Custom average time for line search and ", j, " threads: ", np_total_avg)

def compare_non_linear_cg_method(epoch,  num_of_threads = 16, msize = 1000):
    np.random.seed(3)
    cur_df = np.random.uniform(low=0.5, high=0.7, size=(msize,))
    next_df = np.random.uniform(low=0.5, high=0.7, size=(msize,))
    delta = np.random.uniform(low=0.5, high=0.7, size=(msize,))
    print("case 2")
    np_method = {"Fletcher_Reeves": utils.Fletcher_Reeves_next_iteration, "Dai-Yuan": utils.Fletcher_Reeves_next_iteration, "Hager-Zhang": utils.Fletcher_Reeves_next_iteration}
    naive_method = {
                "Fletcher_Reeves": CG.nonlinear_CG.Naive_Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": CG.nonlinear_CG.Naive_Hager_Zhang_next_iteration,\
                "Dai-Yuan": CG.nonlinear_CG.Naive_Dai_Yuan_next_iteration,\
    }
    accelerated_method = {
                "Fletcher_Reeves": CG.nonlinear_CG.Accelerated_Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": CG.nonlinear_CG.Accelerated_Hager_Zhang_next_iteration,\
                "Dai-Yuan": CG.nonlinear_CG.Accelerated_Dai_Yuan_next_iteration,\
    }
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        sum = 0.0
        method_np = np_method[method]
        for i in range(epoch):
            total = 0.0
            start = time.time()
            _ = method_np(cur_df, next_df, delta)
            end = time.time()
            total += end - start
            sum += total
            #print("Numpy for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Numpy average time for ", method, ": ", np_total_avg)

        sum = 0.0
        method_naive = naive_method[method]
        m_cur_df = Naive_Matrix(cur_df)
        m_next_df = Naive_Matrix(next_df)
        m_delta = Naive_Matrix(delta)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            _ = method_naive(m_cur_df, m_next_df, m_delta)
            end = time.time()
            total += end - start
            sum += total
            #print("Custom for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Custom average time for ", method, ": ", np_total_avg)


        method_accelerated = accelerated_method[method]
        m_cur_df = Accelerated_Matrix(cur_df)
        m_next_df = Accelerated_Matrix(next_df)
        m_delta = Accelerated_Matrix(delta)
        for j in range(2, num_of_threads + 1):
            sum = 0.0
            for i in range(epoch):
                total = 0.0
                start = time.time()
                _ = method_accelerated(m_cur_df, m_next_df, m_delta, j)
                end = time.time()
                total += end - start
                sum += total
                #print("Custom for epoch ", i, " takes ", total, " seconds")
            np_total_avg = sum / epoch
            print("Custom average time for ", method, " and ", j, " threads: ", np_total_avg)

if __name__ == '__main__':
    compare_linear_cg(n = 512, cond = 100000, max_value = 10, epoch = 5, num_of_threads = 4)
    compare_nonlinear_cg_func2(4, 16)
    compare_line_search_func2(4, 16, 1000)
    compare_non_linear_cg_method(4, 16, 1000)