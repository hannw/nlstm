# nlstm
## Tensorflow Implementation of Nested LSTM Cell

Here is a tensorflow implementation of Nested LSTM cell.
It is compatible with the tensorflow rnn API.

```python
from rnn_cell import NLSTMCell
cell = NLSTMCell(num_units=3, depth=2)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
output, new_state = cell(inputs, state=init_state)
...
```

Ref:
- Moniz et al, "Nested LSTMs." https://arxiv.org/abs/1801.10308

