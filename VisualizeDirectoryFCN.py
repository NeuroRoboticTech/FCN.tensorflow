import numpy as np
import FCN
import tensorflow as tf
import os

logs_dir = os.environ['FCN_LOGS']
print("fcn_logs: ", logs_dir)
#if logs_dir == "":
#  print "FCN_LOGS environment variable not found."
#  return

data_dir = os.environ['FCN_DATA']
print("fcn_data: ", data_dir)
#if data_dir == "":
#  print "FCN_DATA environment variable not found."
#  return

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", logs_dir, "path to logs directory") 
tf.flags.DEFINE_string("data_dir", data_dir + "/weeds", "path to dataset") 
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", data_dir + "/model_zoo", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")


def main(argv=None):
  segment = FCN.Segment(False, 672, 380, False)
  segment.init_network(False)

  segment.visualize_directory(
    segment.train_dataset_reader, False, False,
    data_dir + "/weeds/errors/training"
    "F:/Projects/FCN_tensorflow/data/Data_zoo/Weed_errors/errors/training")

  # segment.visualize_directory(
  #   segment.validation_dataset_reader, False, False,
  #   "F:/Projects/FCN_tensorflow/data/Data_zoo/Weed_errors/errors/validation")

  segment.close()

if __name__ == "__main__":
    tf.app.run()
