from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging


import tensorflow as tf
from tensorflow.python.layers import base as base_layer

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class NLSTMCell(rnn_cell_impl.RNNCell):
  """Nested LSTM Cell. Adapted from `rnn_cell_impl.LSTMCell`

  The implementation is based on:
    https://arxiv.org/abs/1801.10308
    JRA. Moniz, D. Krueger.
    "Nested LSTMs"
    ACML, PMLR 77:530-544, 2017
  """

  def __init__(self, num_units, depth, forget_bias=1.0,
               state_is_tuple=True, use_peepholes=False,
               activation=None, gate_activation=None,
               cell_activation=None,
               initializer=None,
               input_gate_initializer=None,
               use_bias=True, reuse=None, name=None):
    """Initialize the basic NLSTM cell.

    Args:
      num_units: `int`, The number of hidden units of each cell state
        and hidden state.
      depth: `int`, The number of layers in the nest.
      forget_bias: `float`, The bias added to forget gates.
      state_is_tuple: If `True`, accepted and returned states are tuples of
        the `h_state` and `c_state`s.  If `False`, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      use_peepholes: `bool`(optional).
      activation: Activation function of the update values,
        including new inputs and new cell states.  Default: `tanh`.
      gate_activation: Activation function of the gates,
        including the input, ouput, and forget gate. Default: `sigmoid`.
      cell_activation: Activation function of the first cell gate. Default: `identity`.
        Note that in the paper only the first cell_activation is identity.
      initializer: Initializer of kernel. Default: `orthogonal_initializer`.
      input_gate_initializer: Initializer of input gates.
        Default: `glorot_normal_initializer`.
      use_bias: `bool`. Default: `True`.
      reuse: `bool`(optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: `str`, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """
    super(NLSTMCell, self).__init__(_reuse=reuse, name=name)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._use_peepholes = use_peepholes
    self._depth = depth
    self._activation = activation or math_ops.tanh
    self._gate_activation = gate_activation or math_ops.sigmoid
    self._cell_activation = cell_activation or array_ops.identity
    self._initializer = initializer or init_ops.orthogonal_initializer()
    self._input_gate_initializer = (input_gate_initializer 
                                    or init_ops.glorot_normal_initializer())
    self._use_bias = use_bias
    self._kernels = None
    self._biases = None
    self.built = False

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple([self._num_units] * (self.depth + 1))
    else:
      return self._num_units * (self.depth + 1)

  @property
  def output_size(self):
    return self._num_units

  @property
  def depth(self):
    return self._depth

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    self._kernels = []
    if self._use_bias:
      self._biases = []

    if self._use_peepholes:
      self._peep_kernels = []
    for i in range(self.depth):
      if i == 0:
        input_kernel = self.add_variable(
            "input_gate_kernel",
            shape=[input_depth, 4 * self._num_units],
            initializer=self._input_gate_initializer)
        hidden_kernel = self.add_variable(
            "hidden_gate_kernel",
            shape=[h_depth, 4 * self._num_units],
            initializer=self._initializer)
        kernel = tf.concat([input_kernel, hidden_kernel],
                           axis=0, name="kernel_0")
        self._kernels.append(kernel)
      else:
        self._kernels.append(
            self.add_variable(
                "kernel_{}".format(i),
                shape=[2 * h_depth, 4 * self._num_units],
                initializer=self._initializer))
      if self._use_bias:
        self._biases.append(
            self.add_variable(
                "bias_{}".format(i),
                shape=[4 * self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype)))
      if self._use_peepholes:
        self._peep_kernels.append(
            self.add_variable(
                "peep_kernel_{}".format(i),
                shape=[h_depth, 3 * self._num_units],
                initializer=self._initializer))

    self.built = True

  def _recurrence(self, inputs, hidden_state, cell_states, depth):
    """use recurrence to traverse the nested structure

    Args:
      inputs: A 2D `Tensor` of [batch_size x input_size] shape.
      hidden_state: A 2D `Tensor` of [batch_size x num_units] shape.
      cell_states: A `list` of 2D `Tensor` of [batch_size x num_units] shape.
      depth: `int`
        the current depth in the nested structure, begins at 0.

    Returns:
      new_h: A 2D `Tensor` of [batch_size x num_units] shape.
        the latest hidden state for current step.
      new_cs: A `list` of 2D `Tensor` of [batch_size x num_units] shape.
        The accumulated cell states for current step.
    """
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    c = cell_states[depth]
    h = hidden_state

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernels[depth])
    if self._use_bias:
      gate_inputs = nn_ops.bias_add(gate_inputs, self._biases[depth])
    if self._use_peepholes:
      peep_gate_inputs = math_ops.matmul(c, self._peep_kernels[depth])
      i_peep, f_peep, o_peep = array_ops.split(
          value=peep_gate_inputs, num_or_size_splits=3, axis=one)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)
    if self._use_peepholes:
      i += i_peep
      f += f_peep
      o += o_peep 

    if self._use_peepholes:
      peep_gate_inputs = math_ops.matmul(c, self._peep_kernels[depth])
      i_peep, f_peep, o_peep = array_ops.split(
          value=peep_gate_inputs, num_or_size_splits=3, axis=one)
      i += i_peep
      f += f_peep
      o += o_peep 

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply

    if self._use_bias:
      forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
      f = add(f, forget_bias_tensor)

    inner_hidden = multiply(c, self._gate_activation(f))

    if depth == 0:
      inner_input = multiply(self._gate_activation(i), self._cell_activation(j))
    else:
      inner_input = multiply(self._gate_activation(i), self._activation(j))

    if depth == (self.depth - 1):
      new_c = add(inner_hidden, inner_input)
      new_cs = [new_c]
    else:
      new_c, new_cs = self._recurrence(
          inputs=inner_input,
          hidden_state=inner_hidden,
          cell_states=cell_states,
          depth=depth + 1)
    new_h = multiply(self._activation(new_c), self._gate_activation(o))
    new_cs = [new_h] + new_cs
    return new_h, new_cs

  def call(self, inputs, state):
    """forward propagation of the cell

    Args:
      inputs: a 2D `Tensor` of [batch_size x input_size] shape
      state: a `tuple` of 2D `Tensor` of [batch_size x num_units] shape
        or a `Tensor` of [batch_size x (num_units * (self.depth + 1))] shape

    Returns:
      outputs: a 2D `Tensor` of [batch_size x num_units] shape
      next_state: a `tuple` of 2D `Tensor` of [batch_size x num_units] shape
        or a `Tensor` of [batch_size x (num_units * (self.depth + 1))] shape
    """
    if not self._state_is_tuple:
      states = array_ops.split(state, self.depth + 1, axis=1)
    else:
      states = state
    hidden_state = states[0]
    cell_states = states[1:]
    outputs, next_state = self._recurrence(inputs, hidden_state, cell_states, 0)
    if self._state_is_tuple:
      next_state = tuple(next_state)
    else:
      next_state = array_ops.concat(next_state, axis=1)
    return outputs, next_state
