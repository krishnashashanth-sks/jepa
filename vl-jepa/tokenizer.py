import torch
import random

class DummyTokenizer:
  def __init__(self, vocab_size, max_seq_len):
    self.vocab_size = vocab_size
    self.max_seq_len = max_seq_len

  def tokenize(self, text=None):
    seq_len = random.randint(1, self.max_seq_len)
    token_ids = torch.randint(1, self.vocab_size, (seq_len,), dtype=torch.long)
    if seq_len < self.max_seq_len:
      padding = torch.zeros(self.max_seq_len - seq_len, dtype=torch.long)
      token_ids = torch.cat([token_ids, padding])
    elif seq_len > self.max_seq_len:
      token_ids = token_ids[:self.max_seq_len]
    return token_ids