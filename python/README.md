# API Introduction
## Overview
 - [Overview](#overview)
 - [Setting Environment](#setting-environment)
 - [Functionality](#functionality)
    - [Linear Conjugate Method](#linear-conjugate-method)
    - [Nonlinear Conjugate Method](#nonlinear-conjugate-method)
## Setting Environment
If you want to use API, please add the ```repo_path/cpp``` 
and ```repo_path/python``` into environment path 
in the python file at first. Then import numpy and 
utils in ```repo_path/python```.

```python
import sys

dir_path = absolute_path_of_the_repo + "/cpp"

sys.path.append(dir_path)

dir_path = absolute_path_of_the_repo + "/python"
sys.path.append(dir_path)

import utils
import numpy
```

## Functionality
### Linear Conjugate Method
Given the quadratic function f(x):

$$\begin{aligned}
f(x) = \frac{1}{2}x^{T}Ax + b^{T}x + c
\end{aligned}
$$

where $A$ is a positive-definite and symmetric matrix.
Then linear CG method can find the global minima ```x_min``` 
by numerical method.

If you want to run the algorithm implemented by numpy, 
use the following function.

```python
x_min = utils.np_linear_CG(x = initial_point, 
                        A = A, 
                        b = b, 
                        epsilon = convergence_tolerance, 
                        epoch = max_iteration)
```

If you want to run the algorithm implemented by c++, 
use the following function.  
When ```use_accelerated = True```, some of the matrix operator would be paralleled.

```python
x_min = utils.custom_linear_CG(x = initial_point, 
                            A = A, 
                            b = b, 
                            epsilon = convergence_tolerance, 
                            epoch = max_iteration,
                            use_accelerated = True,
                            num_threads = number_of_threads)
```

### Nonlinear Conjugate Method
Given the initial point x, convex function f(x) 
and it's derivative function df(x),
the nonlinear CG method can find the local minima ```x_min``` 
by numerical method.

In order to get the df(x) for convenience, I suggest to
use the autograd library.

```python
from autograd import grad
import autograd.numpy as au

def example_function(x):
    """
    Functionality: f(x) = x1^4 + x2^4 + x3^4 + x4^4 ...+ xn^4
    Parameters: 
    x: The input vector.
    """
    x = np.array(x)
    return (np.sum(x**4))**0.5

example_derivative_function = utils.grad(example_function)
```

If you want to run the algorithm implemented by numpy, 
use the following function.

```python
x_min = utils.np_nonlinear_CG(X = initial_point, 
    tol = convergence_tolerance, 
    alpha = hyperparameter_of_convergence_tolerance_in_line_search, 
    beta = convergence_rate_in_line_search, 
    f = function, 
    Df = derivative_function, 
    method = "Fletcher_Reeves")
```

If you want to run the algorithm implemented by c++, 
use the following function.  
When ```use_accelerated = True```, some of the matrix operator would be paralleled.

```python
x_min = utils.custom_nonlinear_CG(X = initial_point, 
    tol = convergence_tolerance, 
    alpha = hyperparameter_of_convergence_tolerance_in_line_search, 
    beta = convergence_rate_in_line_search, 
    f = function, 
    Df = derivative_function, 
    method = "Fletcher_Reeves",
    use_accelerated = True,
    num_threads = number_of_threads)
```