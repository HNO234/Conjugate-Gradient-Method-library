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
    for _ in range(5):
        a = utils.generate_pos_def_symmetric(n, cond, max_value)
        b = np.random.rand(n)
        x = np.random.rand(n)

        sum = 0.0
        count = 0
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_x, count = utils.np_linear_CG(x, a, b, 5e-7)
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
            np_naive_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b, epsilon = 5e-7, epoch=10000, use_accelerated = False)
            end = time.time()
            total += end - start
            sum += total
            #print("Naive for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Naive average time: ", np_total_avg)

        for i in range(1, num_of_threads + 1):
            sum = 0.0
            for j in range(epoch):
                total = 0.0
                start = time.time()
                np_acc_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, epoch=10000, use_accelerated = True, num_threads = i)
                end = time.time()
                total += end - start
                sum += total
                #print("Accelerated for epoch ", j, "and ", i," threads takes ", total, " seconds")
            np_total_avg = sum / epoch
            print("Accelerated average time for ", i, " threads: ", np_total_avg)
        print("Count for this testcase: %d" % count)
     
def compare_nonlinear_cg_func2(epoch,  num_of_threads = 16, n = 1000):
    np.random.seed(3)
    x_rand = np.random.uniform(low=1, high=1, size=(n,))
    print("case 2")
    count = 0
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_x, _, count = utils.np_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method)
            end = time.time()
            total += end - start
            sum += total
            # print("Numpy for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Numpy average time for ", method, "in case 1 : ", np_total_avg)

        sum = 0.0
        x = np.copy(x_rand)
        for i in range(epoch):
            total = 0.0
            start = time.time()
            np_acc_mat_x_min, _ = utils.custom_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method, False, 1)
            end = time.time()
            total += end - start
            sum += total
            #print("Custom for epoch ", i, " takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Custom average time for ", method, "in case 1 : ", np_total_avg)


        for j in range(1, num_of_threads + 1):
            sum = 0.0
            x = np.copy(x_rand)
            for i in range(epoch):
                total = 0.0
                start = time.time()
                np_acc_mat_x_min, _ = utils.custom_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method, True, j)
                end = time.time()
                total += end - start
                sum += total
                #print("Custom for epoch ", i, " takes ", total, " seconds")
            np_total_avg = sum / epoch
            print("Custom average time for ", method, " and ", j, " threads in case 1: ", np_total_avg)
    print("count for this testcase: %d" % count)

if __name__ == '__main__':
    n_list = [256, 512, 1024, 2048, 4096]
    cond_list = [1000, 100000, 10000000]
    for n in n_list:
       for cond in cond_list:
           print("n = %d, cond = %d" % (n, cond))
           compare_linear_cg(n = n, cond = cond, max_value = 10, epoch = 10, num_of_threads = 4)
    n_list = [100, 1000]
    for n in n_list:
        print("n = %d" % n)
        compare_nonlinear_cg_func2(10, 4, n)  