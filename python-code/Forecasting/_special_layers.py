import theano
import theano.tensor as T
import numpy as np

from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple, floatX
from lasagne.random import get_rng
from lasagne.layers.base import Layer, MergeLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "ExpressionLayer",
]

class ExpressionLayer(Layer):
    """
    This layer provides boilerplate for a custom layer that applies a
    simple transformation to the input.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    function : callable
        A function to be applied to the output of the previous layer.
    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.
    Notes
    -----
    An :class:`ExpressionLayer` that does not change the shape of the data
    (i.e., is constructed with the default setting of ``output_shape=None``)
    is functionally equivalent to a :class:`NonlinearityLayer`.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, ExpressionLayer
    >>> l_in = InputLayer((32, 100, 20))
    >>> l1 = ExpressionLayer(l_in, lambda X: X.mean(-1), output_shape='auto')
    >>> l1.output_shape
    (32, 100)
    """
    def __init__(self, incoming, function, output_shape=None, **kwargs):
        super(ExpressionLayer, self).__init__(incoming, **kwargs)

        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.function = function

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            return input_shape
        elif self._output_shape is 'auto':
            input_shape = (0 if s is None else s for s in input_shape)
            X = theano.tensor.alloc(0, *input_shape)
            output_shape = self.function(X).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, input, **kwargs):
        return self.function(input)