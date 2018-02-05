from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import rnn_cell as contrib_rnn_cell
import tensorflow as tf
from tensorflow import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


class TestNLSTM(test.TestCase):

  def _check_tuple_cell(self, *args, **kwargs):
    batch_size = 2
    num_units = 3
    depth = 4
    g = ops.Graph()
    with self.test_session(graph=g) as sess:
      with g.as_default():
        cell = contrib_rnn_cell.NLSTMCell(num_units, depth, *args, **kwargs)
        init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
        output, new_state = cell(
            inputs=random_ops.random_normal([batch_size, 5]),
            state=init_state)
        variables.global_variables_initializer().run()
        vals = sess.run([output, new_state])
    self.assertAllEqual(vals[0], vals[1][0])
    self.assertAllEqual(vals[0].shape, [2, 3])
    for val in vals[1]:
      self.assertAllEqual(val.shape, [2, 3])
    self.assertEqual(len(vals[1]), 5)
    self.assertAllEqual(cell.state_size, [num_units] * (depth + 1))
    self.assertEqual(cell.depth, depth)
    self.assertEqual(cell.output_size, num_units)

  def _check_non_tuple_cell(self, *args, **kwargs):
    batch_size = 2
    num_units = 3
    depth = 2
    g = ops.Graph()
    with self.test_session(graph=g) as sess:
      with g.as_default():
        cell = contrib_rnn_cell.NLSTMCell(num_units, depth,
                                          *args, **kwargs)
        init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
        output, new_state = cell(
            inputs=random_ops.random_normal([batch_size, 5]),
            state=init_state)
        variables.global_variables_initializer().run()
        vals = sess.run([output, new_state])
    self.assertAllEqual(vals[0], vals[1][:, :3])
    self.assertAllEqual(vals[0].shape, [2, 3])
    self.assertAllEqual(vals[1].shape, [2, 9])
    self.assertEqual(cell.state_size, num_units * (depth + 1))
    self.assertEqual(cell.depth, depth)
    self.assertEqual(cell.output_size, num_units)

  def testNLSTMBranches(self):
    state_is_tuples = [True, False]
    use_peepholes = [True, False]
    use_biases = [True, False]
    options = itertools.product(state_is_tuples, use_peepholes, use_biases)
    for option in options:
      state_is_tuple, use_peephole, use_bias = option
      if state_is_tuple:
        self._check_tuple_cell(
            state_is_tuple=state_is_tuple,
            use_peepholes=use_peephole, use_bias=use_bias)
      else:
        self._check_non_tuple_cell(
            state_is_tuple=state_is_tuple,
            use_peepholes=use_peephole, use_bias=use_bias)

