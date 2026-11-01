"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    
    #Initialization
    grad=np.zeros_like(x, dtype=float)

    #Computation
    for idx in np.ndindex(x.shape):
        x[idx]+=eps
        f_plus=f(x)
        x[idx]-=2*eps
        f_minus=f(x)
        x[idx]+=eps
        diff = f_plus-f_minus
        grad[idx]=diff/(2*eps)

    #Result
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    
    #Initialization
    a_grad,n_grad=np.asarray(analytic_grad),np.asarray(numeric_grad)
    diff=np.abs(analytic_grad-numeric_grad)
    
    #Maximum
    max_array=np.maximum(np.maximum(np.abs(a_grad),np.abs(n_grad)),tol)

    #Result
    return np.max(diff/max_array)

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W, b = weight_init_fn(in_dim, out_dim) 
    
    def forward(x):
      y= x @ W +b
      cache = (x,W)
      return (y, cache)

    def backward(dout,cache):
      x,W=cache
      db=np.sum(dout,axis=0)
      dx=dout@(W.T)
      dW= (x.T)@dout
      return (dx, {'W':dW,'b':db})

    dico = {'params' : {'W': W, 'b': b}, 'forward':forward ,  'backward': backward  }

    return dico

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

