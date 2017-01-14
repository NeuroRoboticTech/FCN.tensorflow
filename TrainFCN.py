import numpy as np
import FCN
import tensorflow as tf


def main(argv=None):
  segment = FCN.Segment()
  segment.init_network()
  segment.train_network()

if __name__ == "__main__":
    tf.app.run()
