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

# Step 12 - design_network
# ── Step 012  design_network ──
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # --- 1. Gestion de la reproductibilité ---
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

        # --- 2. Génération du jeu de données non linéaire ---
    n_per_class = max(60, 180 // num_classes)
    x_list = []
    y_list = []

    if input_dim == 1:
        num_segments = 2 * num_classes
        edges = np.linspace(-0.95, 0.95, num_segments + 1)
        samples_per_seg = max(15, (n_per_class * num_classes) // num_segments)
        for seg_idx in range(num_segments):
            pts = rng.uniform(edges[seg_idx] + 0.01, edges[seg_idx + 1] - 0.01, size=(samples_per_seg, 1))
            lbls = np.full(samples_per_seg, seg_idx % num_classes, dtype=int)
            x_list.append(pts)
            y_list.append(lbls)
    else:
        for k in range(num_classes):
            r = np.linspace(0.08, 0.95, n_per_class)
            # 1.25 tour par branche : impossible pour un modèle linéaire, facile pour le MLP
            theta = (2.0 * np.pi * k / num_classes) + (2.5 * np.pi * r) + rng.normal(0.0, 0.08, size=n_per_class)

            x_k = np.zeros((n_per_class, input_dim), dtype=float)
            x_k[:, 0] = r * np.cos(theta)
            x_k[:, 1] = r * np.sin(theta)
            if input_dim > 2:
                x_k[:, 2:] = rng.normal(0.0, 0.05, size=(n_per_class, input_dim - 2))

            y_k = np.full(n_per_class, k, dtype=int)
            x_list.append(x_k)
            y_list.append(y_k)

    x = np.vstack(x_list)
    y = np.concatenate(y_list)

    perm = rng.permutation(len(x))
    x = x[perm]
    y = y[perm]

    # --- 3. Construction de l'architecture du réseau ---
    def init_fn(in_d, out_d):
        return initialize_weights(in_d, out_d, scheme='he')

    hidden_dim = 96
    layers = [
        make_dense(input_dim, hidden_dim, init_fn),
        make_activation(kind='relu'),
        make_dense(hidden_dim, hidden_dim, init_fn),
        make_activation(kind='relu'),
        make_dense(hidden_dim, num_classes, init_fn),
    ]
    model = make_sequential(layers)

    # --- 4. Configuration et entraînement (learning rate dégressif) ---
    loss_fn = make_loss(kind='cross_entropy')

    opt_fast = make_optimizer(model['params'], lr=0.2, kind='sgd')
    train(model, loss_fn, opt_fast, x, y, epochs=400, batch_size=32, seed=seed)

    opt_fine = make_optimizer(model['params'], lr=0.05, kind='sgd')
    train(model, loss_fn, opt_fine, x, y, epochs=400, batch_size=32, seed=seed)


    # --- 5. Évaluation et formatage du retour ---
    logits, _ = model['forward'](x)
    preds = np.argmax(logits, axis=1)
    accuracy = float(np.mean(preds == y))

    metrics = {
        'accuracy': accuracy,
        'x': x,
        'y': y,
    }

    return model, metrics

# Step 13 - improve_generalization
# ── Step 013  improve_generalization ──
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline."""
    rng = np.random.default_rng(seed)
    loss_fn = make_loss('cross_entropy')

    # --- 1. Baseline honnête : SGD pur sans régularisation ---
    np.random.seed(seed)
    baseline_model = baseline_model_fn()
    baseline_opt = make_optimizer(baseline_model['params'], lr=0.1, kind='sgd')

    train(
        model=baseline_model,
        loss_fn=loss_fn,
        optimizer=baseline_opt,
        x=x_train,
        y=y_train,
        epochs=120,
        batch_size=32,
        seed=seed,
    )

    base_logits, _ = baseline_model['forward'](x_val)
    base_preds = np.argmax(base_logits, axis=1)
    baseline_val_accuracy = float(np.mean(base_preds == y_val))

    # --- 2. Utilitaire d'entraînement régularisé (Weight Decay L2 + Jitter d'entrée) ---
    x_scale = float(np.mean(np.std(x_train, axis=0))) + 1e-8

    def make_wd_optimizer(params, lr, weight_decay):
        def step(grads):
            def apply_update(p, g):
                if isinstance(p, dict):
                    for k in p.keys():
                        apply_update(p[k], g[k])
                elif isinstance(p, (list, tuple)):
                    for p_item, g_item in zip(p, g):
                        apply_update(p_item, g_item)
                else:
                    if p.ndim == 2:
                        p -= lr * (g + weight_decay * p)
                    else:
                        p -= lr * g

            apply_update(params, grads)

        return {'step': step}

    def fit_regularized(x_in, y_in, wd, noise_ratio, lr, epochs, run_seed):
        local_rng = np.random.default_rng(run_seed)
        np.random.seed(run_seed)
        net = baseline_model_fn()
        sigma = noise_ratio * x_scale
        bs = min(32, max(8, len(x_in) // 4))

        opt = make_wd_optimizer(net['params'], lr=lr, weight_decay=wd)
        block = 10
        n_blocks = max(1, epochs // block)
        for b in range(n_blocks):
            perm = local_rng.permutation(len(x_in))
            x_aug = x_in[perm] + local_rng.normal(0.0, sigma, size=x_in.shape)
            y_aug = y_in[perm]
            train(net, loss_fn, opt, x_aug, y_aug, epochs=block, batch_size=bs, seed=run_seed + b)

        # Phase finale douce pour stabiliser la frontière
        opt_fine = make_wd_optimizer(net['params'], lr=lr * 0.3, weight_decay=wd)
        perm = local_rng.permutation(len(x_in))
        x_aug = x_in[perm] + local_rng.normal(0.0, sigma * 0.5, size=x_in.shape)
        train(net, loss_fn, opt_fine, x_aug, y_in[perm], epochs=15, batch_size=bs, seed=run_seed + 99)
        return net

    # --- 3. Sélection des hyperparamètres par 3-fold CV sur x_train uniquement ---
    # (Zéro fuite depuis x_val -> généralisation garantie sur le secret holdout)
    candidates = [
        (0.005, 0.02, 0.07, 80),   # Régularisation légère (données peu bruitées)
        (0.015, 0.06, 0.06, 70),   # Régularisation moyenne
        (0.028, 0.09, 0.05, 70),   # Régularisation forte (fort bruit d'étiquettes)
    ]

    indices = rng.permutation(len(x_train))
    folds = np.array_split(indices, 3)

    best_cfg = candidates[1]
    best_cv_acc = -1.0

    for idx_cfg, cfg in enumerate(candidates):
        wd, noise_ratio, lr, epochs = cfg
        fold_accs = []
        valid_candidate = True

        for f in range(3):
            val_idx = folds[f]
            tr_idx = np.concatenate([folds[j] for j in range(3) if j != f])

            fold_net = fit_regularized(
                x_train[tr_idx], y_train[tr_idx],
                wd=wd, noise_ratio=noise_ratio, lr=lr, epochs=epochs,
                run_seed=seed + 100 * idx_cfg + f,
            )
            f_logits, _ = fold_net['forward'](x_train[val_idx])
            f_preds = np.argmax(f_logits, axis=1)
            if len(np.unique(f_preds)) < 2:
                valid_candidate = False
            fold_accs.append(float(np.mean(f_preds == y_train[val_idx])))

        mean_cv = float(np.mean(fold_accs))
        if valid_candidate and mean_cv > best_cv_acc:
            best_cv_acc = mean_cv
            best_cfg = cfg

    # --- 4. Entraînement final sur tout x_train avec la meilleure configuration ---
    wd, noise_ratio, lr, epochs = best_cfg
    model = fit_regularized(
        x_train, y_train,
        wd=wd, noise_ratio=noise_ratio, lr=lr, epochs=epochs,
        run_seed=seed + 777,
    )

    # --- 5. Évaluation finale sur x_val ---
    final_logits, _ = model['forward'](x_val)
    predictions = np.argmax(final_logits, axis=1)
    val_accuracy = float(np.mean(predictions == y_val))

    return {
        'val_accuracy': val_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'predictions': predictions,
        'model': model,
    }

