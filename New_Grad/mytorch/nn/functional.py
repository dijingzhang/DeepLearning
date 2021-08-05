import numpy as np
from mytorch.autograd_engine import Autograd

'''
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
''' 
def add_backward(grad_output, a, b):
    a_grad = grad_output
    b_grad = grad_output
    return a_grad, b_grad

def sub_backward(grad_output, a, b):
    a_grad = grad_output
    b_grad = -grad_output
    return a_grad, b_grad

def matmul_backward(grad_output, a, b):
    a_grad = np.matmul(grad_output, b.transpose())
    b_grad = np.matmul(a.transpose(), grad_output)
    return a_grad, b_grad

def mul_backward(grad_output, a, b):
    a_grad = b * grad_output
    b_grad = a * grad_output
    return a_grad, b_grad

def div_backward(grad_output, a, b):
    a_grad = np.divide(np.ones(b.shape), b) * grad_output
    b_grad = np.divide(-a, np.square(b)) * grad_output
    return a_grad, b_grad

def log_backward(grad_output, a):
    return np.divide(grad_output, a)

def exp_backward(grad_output,a):
    a_grad = grad_output * np.exp(a)
    return a_grad

def max_backward(grad_output, a):
    argmaxes = np.argmax(a, axis=0)
    mask = np.zeros_like(a)
    mask[argmaxes, list(range(a.shape[0]))] = 1
    return grad_output * mask

def sum_backward(grad_output, a):
    mask = np.ones_like(a)
    return mask * grad_output

def relu_backward(grad_output,a):
    a[a>0]=1
    mask = a >= 0
    # out = a*mask
    # autograd.add_operation(input=[mask, a])
    a[a<=0]=0
    return a*grad_output


def softmax(x):
    a = np.max(x)*np.ones(x.shape)
    exp = np.exp(x-a)
    _softmax = exp/exp.sum(1, keepdims=True)
    return _softmax

def SoftmaxCrossEntropy_backward(grad_output, y, y_hat):
    """
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """
    _softmax = softmax(y_hat)
    return None, (_softmax - y)  # *grad_output