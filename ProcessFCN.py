import numpy as np
import FCN
import tensorflow as tf


def main(argv=None):
  fcn = FCN.NetworkFCN()
  fcn.run_train()

if __name__ == "__main__":
    tf.app.run()
