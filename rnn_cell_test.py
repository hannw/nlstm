from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rnn_cell as rc
import tensorflow as tf


class TestNLSTM(tf.test.TestCase):

  def test_nlstm_tuple(self):
    batch_size = 2
    cell = rc.NLSTMCell(num_units=3, depth=4)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output, new_state = cell(inputs=tf.random_normal([batch_size, 5]), state=init_state)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      vals = sess.run([output, new_state])

    self.assertAllEqual(vals[0], vals[1][0])
    self.assertAllEqual(vals[0].shape, [2, 3])
    for val in vals[1]:
      self.assertAllEqual(val.shape, [2, 3])
    self.assertEqual(len(vals[1]), 5)

  def test_nlstm_non_tuple(self):
    batch_size = 2
    cell = rc.NLSTMCell(num_units=3, depth=2, state_is_tuple=False)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output, new_state = cell(inputs=tf.random_normal([batch_size, 5]), state=init_state)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      vals = sess.run([output, new_state])

    self.assertAllEqual(vals[0], vals[1][:, :3])
    self.assertAllEqual(vals[0].shape, [2, 3])
    self.assertAllEqual(vals[1].shape, [2, 9])



