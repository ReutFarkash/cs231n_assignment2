from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        (C, H, W) = input_dim
        F = num_filters

        # conv
        # Input shape : (N, C, H, W)
        # Output shape : (N, F, HH, WW)
        P = (F - 1) / 2
        conv_stride = 1
        HC = 1 + (H + 2 * P - F) / conv_stride
        WC = 1 + (W + 2 * P - F) / conv_stride

        self.params['W1'] = weight_scale * np.random.randn(F, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(F)

        # relu
        # Input shape : (N, F, HC, WC)
        # Output shape : (N, F, HC, WC)
        
        # max pool 2x2
        # Input shape : (N, F, HC, WC)
        # Output shape : (N, F, HP, WP)
        pool_size = 2
        pool_stride = 2
        HP = 1 + (HC - pool_size) / pool_stride
        WP = 1 + (WC - pool_size) / pool_stride

        # affine
        # Input shape : (N, F, HP, WP)
        # Output shape : (N, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(int(F * HP * WP), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        #relu
        # Input shape : (N, hidden_dim)
        # Output shape : (N, hidden_dim)

        # affine
        # Input shape : (N, hidden_dim)
        # Output shape : (N, num_classes)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        # softmax

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        c1, c1_cache = conv_forward_fast(X, W1, b1, conv_param)
        r1, r1_cache = relu_forward(c1)
        p1, p1_cache = max_pool_forward_fast(r1, pool_param)
        a1, a1_cache = affine_forward(p1, W2, b2)
        r2, r2_cache = relu_forward(a1)
        a2, a2_cache = affine_forward(r2, W3, b3)
        scores = a2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        reg = self.reg
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5*reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        dr2, grads['W3'], grads['b3'] = affine_backward(dscores, a2_cache)
        da1 = relu_backward(dr2, r2_cache)
        dp1, grads['W2'], grads['b2'] = affine_backward(da1, a1_cache)
        dr1 = max_pool_backward_fast(dp1, p1_cache)
        dc1 = relu_backward(dr1, r1_cache)
        dX, grads['W1'], grads['b1'] = conv_backward_fast(dc1, c1_cache)
        grads['W1'] += reg*W1
        grads['W2'] += reg*W2
        grads['W3'] += reg*W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
