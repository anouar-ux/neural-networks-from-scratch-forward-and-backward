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

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    
    def forward(x):
      result=x.copy()       # Keep x unchanged
      result[result<0]=0    # ReLU activation
      return (result,x)

    def backward(dout,cache):
      dx= dout* (cache>0)   # dx[i]=dout iff x>0, else 0
      return (dx, {} )

    return {'forward':forward, 'backward':backward, 'params':{}}

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    
    s= (2/in_dim)**0.5
    W = np.random.randn(in_dim, out_dim) * s
    b= np.zeros(out_dim)

    return (W,b)

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    
    def loss_fn(logits,labels):
      batch,C=logits.shape

      stable_logits=logits.copy()
      stable_logits-=np.max(stable_logits,axis=1, keepdims=True)

      softmax = np.exp(stable_logits)/np.sum(np.exp(stable_logits),axis=1, keepdims=True)

      mask= np.zeros_like(logits, dtype=float)
      mask[np.arange(batch), labels] = 1

      correct= softmax[np.arange(batch), labels]
      loss= -np.sum(np.log(correct))/batch
      backward = (softmax-mask)/batch

      return (loss, backward)

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    params=[layers[k]['params'] for k in range(len(layers))]

    def forward(x):
      caches=[]
      for dico in layers:
        x_new,cache=dico['forward'](x)
        x=x_new
        caches.append(cache)
      return (x, caches)

    def backward(dout,caches):
      error = dout.copy()
      grads_list =[]
      for k in range(len(layers)-1,-1,-1):
        cache = caches[k]
        dx,grad= layers[k]['backward'](error,cache)
        error=dx
        grads_list.append(grad)
      grads_list.reverse()

      return (error, grads_list)

    return {'params':params, 'forward':forward, 'backward':backward}

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    
    logits, caches = model['forward'](x)

    loss,d_logits= loss_fn(logits, y)

    dx, param_grads =model['backward'](d_logits, caches)

    return (loss,param_grads)

# Step 9 - make_optimizer
# ── Step 009  make_optimizer ──
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. 
    """
    
    def step(grads):
        # Fonction interne récursive pour parcourir n'importe quelle structure
        def apply_update(p, g):
            if isinstance(p, dict):
                for k in p.keys():
                    apply_update(p[k], g[k])
            elif isinstance(p, (list, tuple)):
                for p_item, g_item in zip(p, g):
                    apply_update(p_item, g_item)
            else:
                # Arrivé au tableau NumPy : mise à jour in-place grâce au -=
                p -= lr * g
                
        # On lance la mise à jour sur la structure globale
        apply_update(params, grads)

    return {'step': step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    
    loss, param_grads = forward_backward(model, loss_fn, x_batch,y_batch)

    step= optimizer['step']
    step(param_grads)

    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    history = []

    for t in range(epochs):
      S,m=0,0
      N= x.shape[0]
      while (m+1)*batch_size<N:
        x_batch,y_batch = x[m*batch_size:(m+1)*batch_size][:],y[m*batch_size:(m+1)*batch_size][:]
        loss = train_step(model, loss_fn, optimizer, x_batch,y_batch)
        S+= loss
        m+=1
      if m*batch_size<N:
        x_batch,y_batch = x[m*batch_size:][:],y[m*batch_size:][:]
        loss = train_step(model, loss_fn, optimizer, x_batch,y_batch)
        S+= loss
        m+=1
      if m== 0:
        return "x batch is empty"
      else:
        history.append(S/m)

    return history

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

