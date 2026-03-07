# Part II Comparable Run Summary

| Model | Embedding | Epochs | Best Val PPL | Final Val PPL | Mean Epoch Time (s) | Coverage | Log |
|-------|-----------|--------|--------------|---------------|---------------------|----------|-----|
| RNN | Trainable | 10 | 183.2731 | 183.2731 | 11.03 | — | `rnn_trainable_20260307_021319.log` |
| RNN | Word2Vec | 10 | 165.9518 | 165.9518 | 10.58 | — | `rnn_fixed_self_20260307_022509.log` |
| RNN | GloVe | 10 | 187.0129 | 187.0129 | 10.56 | 0.9199 | `rnn_fixed_pretrained_20260307_022954.log` |
| LSTM | Trainable | 10 | 336.5799 | 336.5799 | 12.13 | — | `lstm_trainable_20260307_023450.log` |
| LSTM | Word2Vec | 10 | 304.2055 | 304.2055 | 11.74 | — | `lstm_fixed_self_20260307_023955.log` |
| LSTM | GloVe | 10 | 403.4921 | 403.4921 | 11.69 | 0.9199 | `lstm_fixed_pretrained_20260307_024458.log` |
| TRANSFORMER | Trainable | 10 | 213.4547 | 213.4547 | 13.33 | — | `transformer_trainable_20260307_025011.log` |
| TRANSFORMER | Word2Vec | 10 | 174.4335 | 174.4335 | 13.20 | — | `transformer_fixed_self_20260307_025526.log` |
| TRANSFORMER | GloVe | 10 | 187.1095 | 187.1095 | 13.21 | 0.9199 | `transformer_fixed_pretrained_20260307_030039.log` |

## Missing Rows

- None

## Incomplete Rows (< 10 epochs)

- None
