"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix,axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix,axis=1,keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix,axis=1,keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    row_m=row_max(logits)
    ulogits=logits-row_m
    return np.exp(ulogits)

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    stable_log=exp_shifted(logits)
    tot=row_sum(stable_log)
    return stable_log/tot

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    return np.eye(num_classes)[labels]

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    
    return np.diag(probs[:,labels])

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    based=gather_true_class_probs(probs, labels).clip(eps)
    return -np.mean(np.log(based)).item()

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    return np.sum(argmax_rows(logits_or_probs)==labels)/len(labels)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    sigma=he_std(fan_in)
    return np.random.normal(0,sigma,shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length,dtype='float64')

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    N,C,H,W=images.shape
    img=np.zeros((N,C,H+2*pad,W+2*pad),dtype=images.dtype)
    start_row=pad 
    end_row=pad+H
    start_col=pad 
    end_col=pad+W 
    img[:,:,start_row:end_row,start_col:end_col]=images[:,:,:,:]
    return img

# Step 14 - output_spatial_size
import math
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return (input_size-kernel+2*padding)//stride + 1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    N,C,H,W=images.shape
    out_h=output_spatial_size(H, kernel_h, stride, padding)
    out_w=output_spatial_size(W, kernel_w, stride, padding)
    conv=np.zeros((N * out_h * out_w, C * kernel_h * kernel_w),dtype=images.dtype)
    img=pad_2d(images, padding)
    k=0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                start_i = i * stride
                start_j = j * stride
                conv[k]=img[n, :, start_i:start_i + kernel_h,start_j:start_j + kernel_w].flatten()
                k+=1
    return conv

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    (N, C, H, W)=input_shape
    out_h=output_spatial_size(H, kernel_h, stride, padding)
    out_w=output_spatial_size(W, kernel_w, stride, padding)
    image=np.zeros(input_shape)
    img=pad_2d(image, padding)
    k=0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                start_i = i * stride
                start_j = j * stride
                img[n, :, start_i:start_i + kernel_h,start_j:start_j + kernel_w]+=cols[k].reshape(-1,kernel_h,kernel_w)
                k+=1
    if padding>0:
        return img[:,:,padding:-padding,padding:-padding]
    return img

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    N,C,H,W=x.shape  
    out_h=output_spatial_size(H, weights.shape[-2], stride, padding)
    out_w=output_spatial_size(W, weights.shape[-1], stride, padding)
    x_cols=im2col(x, weights.shape[-2], weights.shape[-1], stride, padding)
    Y=x_cols@ weights.reshape(weights.shape[0],-1).T + bias 
    D={'cols':x_cols, 'kernel_h':weights.shape[-2], 'kernel_w':weights.shape[-1],
     'padding':padding, 'stride':stride, 'weights':weights, 'x_shape':x.shape}
    return Y.reshape(N,out_h,out_w,weights.shape[0]).transpose(0, 3, 1, 2),D

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']
    padding = cache['padding']
    stride = cache['stride']
    weights = cache['weights']
    x_shape = cache['x_shape']

    C_out = weights.shape[0]

    # Match the row ordering produced by im2col:
    # sample -> output row -> output column
    d_out_cols = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    # (C_out, C_in * kernel_h * kernel_w)
    W_col = weights.reshape(C_out, -1)

    # (N*out_h*out_w, C_in*kernel_h*kernel_w)
    d_cols = d_out_cols @ W_col

    # Fold the gradient patches back into input layout
    dx = col2im(
        d_cols,
        x_shape,
        kernel_h,
        kernel_w,
        stride,
        padding
    )
    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    cols = cache['cols']
    weights = cache['weights']
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']

    C_out = weights.shape[0]

    # Match the matrix layout used in conv2d_forward
    d_Y = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    # Gradient with respect to flattened weights
    d_weights_flat = d_Y.T @ cols

    # Restore original weight shape
    d_weights = d_weights_flat.reshape(
        C_out,
        weights.shape[1],
        kernel_h,
        kernel_w
    )

    return d_weights

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    x=np.sum(d_out,axis=(0,2,3))
    return x

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx=conv2d_grad_input(d_out, cache)
    dw=conv2d_grad_weights(d_out, cache)
    db=conv2d_grad_bias(d_out)
    return (dx, dw, db)

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N,C,H,W=x.shape
    out_h=output_spatial_size(H, kernel, stride, 0)
    out_w=output_spatial_size(W, kernel, stride, 0)
    out=np.zeros((N,C,out_h,out_w),dtype=x.dtype)
    argmax=np.zeros((N,C,out_h,out_w),dtype='int64')
    for i in range(N):
        for j in range(C):
            for k in range(out_h):
                for l in range(out_w):
                    start_r=k*stride
                    start_c=l*stride
                    window=x[i,j,start_r:start_r + kernel,start_c:start_c + kernel]
                    flat = window.reshape(-1)
                    idx=np.argmax(flat)
                    out[i,j,k,l]=flat[idx]
                    argmax[i,j,k,l]=idx
    cache = {
    'x_shape': x.shape,
    'argmax': argmax,
    'kernel': kernel,
    'stride': stride }
    return out,cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    row=argmax_index//kernel 
    col=argmax_index%kernel 
    z=np.zeros((kernel,kernel))
    z[row,col]=grad_value
    return z

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    kernel= cache['kernel']
    stride = cache['stride']
    argmax = cache['argmax']
    x_shape = cache['x_shape']
    N,C,H,W=x_shape
    out_h=d_out.shape[2]
    out_w=d_out.shape[3]
    out=np.zeros(x_shape)
    for i in range(N):
        for j in range(C):
            for k in range(out_h):
                for l in range(out_w):
                    start_r=k*stride
                    start_c=l*stride
                    grad=d_out[i,j,k,l]
                    argmax_index=argmax[i,j,k,l]
                    grad_window=scatter_grad_window(grad, argmax_index, kernel)
                    out[i,j,start_r:start_r+kernel,start_c:start_c+kernel]+=grad_window
    return out

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    cache={'x':x}
    return np.maximum(x,0),cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    mask=np.where(cache['x']>0.0,1.0,0.0)
    return mask*d_out

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    out=x.reshape(x.shape[0],-1)
    c={'x_shape':x.shape}
    return (out,c)

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    return d_out.reshape(cache['x_shape'])

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    return x@ weights+bias, {'x':x, 'weights':weights}

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    return d_out@ cache['weights'].T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    return x.T@ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    return np.sum(dout,axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    return linear_grad_input(dout, cache),linear_grad_weights(cache['x'], dout),linear_grad_bias(dout)

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    probs= stable_softmax(logits)
    return np.abs(cross_entropy_loss(probs, y))

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    probs=stable_softmax(logits)
    indicator=one_hot(y, logits.shape[1])
    return (probs-indicator)/len(y)

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    return param -lr*grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    return beta_one*m +(1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    return beta_two*v +(1-beta_two)*(grad**2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return (moment/(1- beta**t))

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    return param - lr * m_hat/((v_hat)**0.5+eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    new_m=adam_update_m(m, grad, beta_one)
    new_v=adam_update_v(v, grad, beta_two)
    m_hat=adam_bias_correct(new_m, beta_one, t)
    v_hat=adam_bias_correct(new_v, beta_two, t)
    new_param=adam_param_step(param, m_hat, v_hat, lr, eps)
    return (new_param, new_m, new_v)

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    x=in_channels*kernel_size*kernel_size
    w=he_init((out_channels, in_channels, kernel_size, kernel_size), x, seed)
    b=init_zero_bias(out_channels)
    return {'W':w,"b":b}

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    b= init_zero_bias(out_features)
    w= he_init((in_features, out_features), in_features, seed)
    return {'W':w,"b":b}

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    conv1=init_conv_layer(6, in_channels, 5, seed)
    conv2=init_conv_layer(16,6, 5, seed)
    fc1=init_linear_layer(16*4*4, 120, seed)
    fc2=init_linear_layer(120, num_classes, seed)
    return {'conv1':conv1, 'conv2':conv2, 'fc1':fc1, 'fc2':fc2}

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    Y,cache=conv2d_forward(x, W, b, stride, pad)
    Y,relu_cache=relu_forward(Y)
    pool,pool_cache=maxpool2d_forward(Y, pool_size, pool_size)
    cache_dict={'conv_cache':cache,
                'relu_cache':relu_cache,
                'pool_cache':pool_cache}
    return pool,cache_dict

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    out,cache_flat=flatten_forward(x)
    out,fc1_cache=linear_forward(out, fc1['W'], fc1['b'])
    out,relu_cache=relu_forward(out)
    out,fc2_cache=linear_forward(out, fc2['W'], fc2['b'])
    return out,{ 'flatten_cache':cache_flat,
                 'fc1_cache':fc1_cache,
                 'relu_cache':relu_cache,
                 'fc2_cache':fc2_cache}

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    conv1=params['conv1']
    y,block1=forward_conv_block(x, conv1['W'], conv1['b'], 2, 1, 0)
    conv2=params['conv2']
    y,block2=forward_conv_block(y, conv2['W'], conv2['b'], 2, 1, 0)
    fc1=params['fc1']
    fc2=params['fc2']
    out,classifier=forward_classifier_block(y, fc1, fc2)
    return out,{'block1':block1, 'block2':block2,'classifier':classifier}

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    dout=maxpool2d_backward(dout, cache['pool_cache'])
    dout=relu_backward(dout, cache['relu_cache'])
    return conv2d_backward(dout, cache['conv_cache'])

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values
    D={}
    dx,dw,db=linear_backward(dlogits, cache['fc2_cache'])
    D['fc2']={'dW':dw,'db':db}
    dx=relu_backward(dx, cache['relu_cache'])
    dx,dw,db=linear_backward(dx, cache['fc1_cache'])
    D['fc1']={'dW':dw,'db':db}
    D['dx']=flatten_backward(dx, cache['flatten_cache'])
    return D

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    D=backward_classifier_block(dlogits, caches['classifier'])
    C={}
    C['fc1']=D['fc1']
    C['fc2']=D['fc2']
    dx1,dw1,db1=backward_conv_block(D['dx'], caches['block2'])
    C['conv2']={'dW':dw1,'db':db1}
    dx2,dw2,db2=backward_conv_block(dx1, caches['block1'])
    C['conv1']={'dW':dw2,'db':db2}
    return C

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    y,_=lenet_forward(x, params)
    return np.argmax(y,axis=1)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    rng=np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=num_samples)
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    shift = y - (num_classes - 1) / 2
    x += shift[:, None, None, None]
    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).
    np.random.seed(seed)
    index=np.arange(n)
    np.random.shuffle(index)
    return index

# Step 54 - train_test_split
import math
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.
    n=len(y)
    index=shuffle_indices(n, seed)
    end_train=math.ceil(n*(1-test_fraction))
    X=x[index]
    x_train=X[:end_train]
    label=y[index]
    y_train=label[:end_train]
    x_test=X[end_train:]
    y_test=label[end_train:]
    return (x_train, y_train, x_test, y_test)

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    # TODO: yield shuffled mini-batches of features and labels for one epoch of training.
    N=len(y)
    indices = shuffle_indices(N, seed=seed)

    for start in range(0, N - N % batch_size, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        xb = x[batch_indices]
        yb = y[batch_indices]
        yield xb, yb

# Step 56 - train_step
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    # TODO: Run forward + loss + backward and apply one Adam update to every parameter.
    logits, cache = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dlogits = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dlogits, cache)
    new_params = {}
    new_opt_state = {}
    for layer, layer_params in params.items():
        new_params[layer] = {}
        new_opt_state[layer] = {}
        for pname, param in layer_params.items():
            if pname == 'W':
                grad = grads[layer]['dW']
            elif pname == 'b':
                grad = grads[layer]['db']
            else:
                continue
            m = opt_state[layer][pname]['m']
            v = opt_state[layer][pname]['v']

            new_param, new_m, new_v = adam_step(
                param,
                grad,
                m,
                v,
                step,
                lr,
                beta_one,
                beta_two,
                eps
            )

            new_params[layer][pname] = new_param
            new_opt_state[layer][pname] = {
                'm': new_m,
                'v': new_v
            }

    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    # TODO: iterate minibatches and apply one train_step per batch, tracking losses and step_counter.
    losses = []
    for xb, yb in iterate_minibatches(x, y, batch_size, seed=seed):
        step_counter += 1
        params, opt_state, loss = train_step(
            params,
            opt_state,
            xb,
            yb,
            lr,
            beta_one,
            beta_two,
            eps,
            step_counter)
        losses.append(loss)
    return params, opt_state, step_counter, losses

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

