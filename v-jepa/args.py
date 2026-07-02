import argparse
class Args:
  def __init__(self):
    self.word_size=1
    self.rank=0
    self.gpu=0
    self.dist_backend='nccl'
    self.dist_url='env://'