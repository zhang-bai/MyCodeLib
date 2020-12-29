"""
神经网络层级结构
1.图卷积层
    1)Graphconv_Pytorch
    2)Graphconv_Tensorflow
"""


def Graphconv_Pytorch():
    import math
    import torch
    from torch import nn
    from torch.nn.parameter import Parameter

    class SparseMM(torch.autograd.Function):

        def __init__(self, sparse):
            super(SparseMM, self).__init__()
            self.sparse = sparse

        def forward(self, dense):
            return torch.mm(self.sparse, dense)

        def backward(self, grad_output):
            grad_input = None
            if self.needs_input_grad[0]:
                grad_input = torch.mm(self.sparse.t(), grad_output)
            return grad_input


    class GraphConv_Pytorch(nn.Module):

        def __init__(self, opt):
            super(GraphConv_Pytorch, self).__init__()
            self.opt = opt
            self.in_size = opt['in']
            self.out_size = opt['out']
            # self.adj = adj
            self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.out_size)
            self.weight.data.uniform_(-stdv, stdv)

        def forward(self, x, adj):
            m = torch.mm(x, self.weight)
            m = SparseMM(adj)(m)
            return m

        def forward_aux(self,x):
            m = torch.mm(x, self.weight)
            return m


def Graphconv_Tensorflow():
    import tensorflow as tf
    import numpy as np

    def uniform(shape, scale=0.05, name=None):
        """Uniform init."""
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def glorot(shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def zeros(shape, name=None):
        """All zeros."""
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def ones(shape, name=None):
        """All ones."""
        initial = tf.ones(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # global unique layer ID dictionary for layer name assignment
    _LAYER_UIDS = {}

    def get_layer_uid(layer_name=''):
        """Helper function, assigns unique layer IDs."""
        if layer_name not in _LAYER_UIDS:
            _LAYER_UIDS[layer_name] = 1
            return 1
        else:
            _LAYER_UIDS[layer_name] += 1
            return _LAYER_UIDS[layer_name]

    def sparse_dropout(x, keep_prob, noise_shape):
        """Dropout for sparse tensors."""
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(x, dropout_mask)
        return pre_out * (1. / keep_prob)

    def dot(x, y, sparse=False):
        """Wrapper for tf.matmul (sparse vs dense)."""
        if sparse:
            res = tf.sparse_tensor_dense_matmul(x, y)
        else:
            res = tf.matmul(x, y)
        return res

    class Layer(object):
        """Base layer class. Defines basic API for all layer objects.
        Implementation inspired by keras (http://keras.io).

        # Properties
            name: String, defines the variable scope of the layer.
            logging: Boolean, switches Tensorflow histogram logging on/off

        # Methods
            _call(inputs): Defines computation graph of layer
                (i.e. takes input, returns output)
            __call__(inputs): Wrapper for _call()
            _log_vars(): Log all variables
        """

        def __init__(self, **kwargs):
            allowed_kwargs = {'name', 'logging'}
            for kwarg in kwargs.keys():
                assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
            name = kwargs.get('name')
            if not name:
                layer = self.__class__.__name__.lower()
                name = layer + '_' + str(get_layer_uid(layer))
            self.name = name
            self.vars = {}
            logging = kwargs.get('logging', False)
            self.logging = logging
            self.sparse_inputs = False

        def _call(self, inputs):
            return inputs

        def __call__(self, inputs):
            with tf.name_scope(self.name):
                if self.logging and not self.sparse_inputs:
                    tf.summary.histogram(self.name + '/inputs', inputs)
                outputs = self._call(inputs)
                if self.logging:
                    tf.summary.histogram(self.name + '/outputs', outputs)
                return outputs

        def _log_vars(self):
            for var in self.vars:
                tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

    class Dense(Layer):
        """Dense layer."""

        def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                     act=tf.nn.relu, bias=False, featureless=False, **kwargs):
            super(Dense, self).__init__(**kwargs)

            if dropout:
                self.dropout = placeholders['dropout']
            else:
                self.dropout = 0.

            self.act = act
            self.sparse_inputs = sparse_inputs
            self.featureless = featureless
            self.bias = bias

            # helper variable for sparse dropout
            self.num_features_nonzero = placeholders['num_features_nonzero']

            with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = glorot([input_dim, output_dim],
                                              name='weights')
                if self.bias:
                    self.vars['bias'] = zeros([output_dim], name='bias')

            if self.logging:
                self._log_vars()

        def _call(self, inputs):
            x = inputs

            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

            # transform
            output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

            # bias
            if self.bias:
                output += self.vars['bias']

            return self.act(output)

    class GraphConvolution(Layer):
        """Graph convolution layer."""

        def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                     sparse_inputs=False, act=tf.nn.relu, bias=False,
                     featureless=False, **kwargs):
            super(GraphConvolution, self).__init__(**kwargs)

            if dropout:
                self.dropout = placeholders['dropout']
            else:
                self.dropout = 0.

            self.act = act
            self.support = placeholders['support']
            self.sparse_inputs = sparse_inputs
            self.featureless = featureless
            self.bias = bias

            # helper variable for sparse dropout
            self.num_features_nonzero = placeholders['num_features_nonzero']

            with tf.variable_scope(self.name + '_vars'):
                for i in range(len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
                if self.bias:
                    self.vars['bias'] = zeros([output_dim], name='bias')

            if self.logging:
                self._log_vars()

        def _call(self, inputs):
            x = inputs

            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

            # convolve
            supports = list()
            for i in range(len(self.support)):
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)],
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)

            # bias
            if self.bias:
                output += self.vars['bias']

            return self.act(output)
