import numpy as np
import FCN
import tensorflow as tf

tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "F:/tmp/FCN/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "F:/Projects/FCN_tensorflow/data/Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def main(argv=None):
  segment = FCN.Segment()
  segment.init_network()
  segment.train_network()

if __name__ == "__main__":
    tf.app.run()
