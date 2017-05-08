import numpy as np
import FCN
import tensorflow as tf
import sys
import FCN_env_vars as EnvVars

tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", EnvVars.logs_dir, "path to logs directory")
tf.flags.DEFINE_string("data_dir", EnvVars.data_dir + "/Data_Zoo/Humans", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", EnvVars.data_dir + "/Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def main(argv=None):
  # try:
  # segment = FCN.Segment(False, 512, 340)
  segment = FCN.Segment(True, 224, 224)
  segment.init_network(True)
  segment.train_network()
  segment.close()
  # except:
  #   e = sys.exc_info()
  #   print("Unexpected error:", e)


if __name__ == "__main__":
    tf.app.run()
