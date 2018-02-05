from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rnn_cell as contrib_rnn_cell
import tensorflow as tf
from tensorflow import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


class TestNLSTM(test.TestCase):

  def testNLSTMTuple(self):
    batch_size = 2
    num_units = 3
    depth = 4
    cell = contrib_rnn_cell.NLSTMCell(num_units=num_units, depth=depth)
    init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
    output, new_state = cell(
        inputs=random_ops.random_normal([batch_size, 5]), state=init_state)

    with self.test_session() as sess:
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

  def testNLSTMNonTuple(self):
    batch_size = 2
    num_units = 3
    depth = 2
    cell = contrib_rnn_cell.NLSTMCell(
        num_units=num_units, depth=depth, state_is_tuple=False)
    init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
    output, new_state = cell(
        inputs=random_ops.random_normal([batch_size, 5]), state=init_state)

    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      vals = sess.run([output, new_state])

    self.assertAllEqual(vals[0], vals[1][:, :3])
    self.assertAllEqual(vals[0].shape, [2, 3])
    self.assertAllEqual(vals[1].shape, [2, 9])
    self.assertEqual(cell.state_size, num_units * (depth + 1))
    self.assertEqual(cell.depth, depth)
    self.assertEqual(cell.output_size, num_units)

  def testNLSTMNoBias(self):
    batch_size = 2
    num_units = 3
    depth = 4
    cell = contrib_rnn_cell.NLSTMCell(
        num_units=num_units, depth=depth, use_bias=False)
    init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
    output, new_state = cell(
        inputs=random_ops.random_normal([batch_size, 5]), state=init_state)

    with self.test_session() as sess:
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

  def testNLSTMNoBiasNonTuple(self):
    batch_size = 2
    num_units = 3
    depth = 2
    cell = contrib_rnn_cell.NLSTMCell(
        num_units=num_units, depth=depth,
        state_is_tuple=False, use_bias=False)
    init_state = cell.zero_state(batch_size, dtype=dtypes.float32)
    output, new_state = cell(
        inputs=random_ops.random_normal([batch_size, 5]), state=init_state)

    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      vals = sess.run([output, new_state])

    self.assertAllEqual(vals[0], vals[1][:, :3])
    self.assertAllEqual(vals[0].shape, [2, 3])
    self.assertAllEqual(vals[1].shape, [2, 9])
    self.assertEqual(cell.state_size, num_units * (depth + 1))
    self.assertEqual(cell.depth, depth)
    self.assertEqual(cell.output_size, num_units)


