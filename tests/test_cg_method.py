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


def test_linear_cg():
    np.random.seed(0)
    a = utils.generate_pos_def_symmetric(100, 1000, 10)
    b = np.random.rand(100)
    x = np.random.rand(100)
    x_np, _ = utils.np_linear_CG(x, a, b, 5e-7)
    print('x',x_np)
    # Test the numpy linear CG method
    x_min = np.linalg.solve(a, b) # Differentiate to find the minimizer
    print('x_min', x_min)
    assert(np.isclose(x_min, x_np).all())
    # Test the custom linear CG method without acceleration
    np_naive_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, use_accelerated = False)
    print('np_naive_mat_x_min', np_naive_mat_x_min)
    assert(np.isclose(x_min, np_naive_mat_x_min).all())
    # Test the custom linear CG method with acceleration
    np_acc_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, epoch=10000, use_accelerated = True)
    print(' np_acc_mat_x_min', np_acc_mat_x_min)
    assert(np.isclose(x_min, np_acc_mat_x_min).all())

def test_nonlinear_cg():
    np.random.seed(3)
    x_rand = np.random.uniform(low=3, high=5, size=(2,))
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        x = np.copy(x_rand)
        print("case 1 np: ", method)
        x, _ = utils.np_nonlinear_CG(x, 5e-8, 0.5, 0.8, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method)
        x = np.array(x)
        x_min = np.array([1, 1])
        assert(np.isclose(x_min, x).all())

        print("case 1 custom : ", method," , without acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(x, 5e-8, 0.5, 0.8, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method, use_accelerated = False, num_threads = 1)
        x = np.array(x)
        x_min = np.array([1, 1])
        assert(np.isclose(x_min, x).all())

        print("case 1 custom : ", method," , with acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(x, 5e-8, 0.5, 0.8, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method, use_accelerated = True, num_threads = 2)
        x = np.array(x)
        x_min = np.array([1, 1])
        assert(np.isclose(x_min, x).all())              

    x_rand = np.random.uniform(low=0.5, high=0.7, size=(100,))
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        print("case 2 np: ", method)
        x = np.copy(x_rand)
        x, _ = utils.np_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method)
        x = np.array(x)
        x_min = np.zeros(100)
        assert(np.isclose(x_min, x).all())            

        print("case 2 custom: ", method," , without acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method, use_accelerated = False, num_threads = 1)
        x = np.array(x)
        x_min = np.zeros(100)
        assert(np.isclose(x_min, x).all())   

        print("case 2 custom: ", method," , with acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(x, 1e-8, 0.5, 0.8, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method, use_accelerated = True, num_threads = 2)
        x = np.array(x)
        x_min = np.zeros(100)
        assert(np.isclose(x_min, x).all())   

if __name__ == '__main__':
    test_linear_cg()
    test_nonlinear_cg()