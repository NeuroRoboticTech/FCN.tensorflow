import numpy as np
import FCN
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "F:/tmp/FCN/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "F:/Projects/FCN_tensorflow/data/Data_zoo/Weed_Errors/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "F:/Projects/FCN_tensorflow/data/Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")


def main(argv=None):
  segment = FCN.Segment(True)
  segment.init_network(False)

  segment.visualize_directory(
    segment.train_dataset_reader, False, False,
    "F:/Projects/FCN_tensorflow/data/Data_zoo/Weed_errors/errors/training")

  # segment.visualize_directory(
  #   segment.validation_dataset_reader, False, False,
  #   "F:/Projects/FCN_tensorflow/data/Data_zoo/Weed_errors/errors/validation")

  segment.close()

if __name__ == "__main__":
    tf.app.run()
