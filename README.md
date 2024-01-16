# Conjugate Gradient Method Library
Conjugate Gradient Method Library implement the linear CG method and 
non-linear CG menthod by numpy and C++. 
Furthermore, Conjugate Gradient Method Library supports OpenMP and CUDA operations, that can further accelerate the C++ library.

## Introduction to the Conjugate Gradient Method
Conjugate gradient method is a numerical method that can find the 
minima of the function in the hyper-dimensional space, and conjugate 
gradient method includes linear conjugate method and non-linear conjugate method. 

Linear CG algorithm can precisely calculate the step length in every iteration,
but in order to calculate the precise step length, the objective function could
only be quadratic function. Compared with the steepest gradient method, the 
minima could be found in the 
finite step by conjugate gradient method in theory (if we don't consider the 
floating-point error and ill-conditioned), and the convergence iteration of 
conjugate gradient method is less than those of steepest gradient method.

Non-linear CG algorithm can approximately calculate the step length by line 
search, or gradient descent method, and the advantage of non-linear CG 
algorithm is that the target function could be convex nonlinear objective 
functions.

## Getting Started

1. Clone this repo.

2. Install [numpy](https://numpy.org/install/), 
[autograd](https://github.com/HIPS/autograd), 
[OpenMP](https://www.openmp.org/), 
and [CUDA](https://developer.nvidia.com/cuda-toolkit), 
or build and run the dockerfile in /contrib/docker (under construction).

3. Here are the commands for the CPU implementations:

| Command | Utility | 
| :----: | :----: |
| ```make``` | Compile the module |
| ```make test``` | Run the pytest |
| ```make demo``` | Run the simple example and analysis. |

For the GPU implementations, simply add the ```GPU=1``` flag:

| Command | Utility | 
| :----: | :----: |
| ```make GPU=1``` | Compile the module |
| ```make test GPU=1``` | Run the pytest |
| ```make demo GPU=1``` | Run the simple example and analysis. |

Run ```python3 demo/demo_cg_method.py``` for reproducing the results in the paper (which is also included in ```make demo```). 

## User Tutorial
<a href="./python">API Introduction</a>

