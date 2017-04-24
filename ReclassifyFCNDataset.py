import numpy as np
import FCN
import tensorflow as tf
import FCN_env_vars as EnvVars

project_Dir = "weeds"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", EnvVars.logs_dir, "path to logs directory")
tf.flags.DEFINE_string("data_dir", EnvVars.data_dir + "/" + project_Dir, "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", EnvVars.data_dir + "/model_zoo", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")


def main(argv=None):
  # segment = FCN.Segment(False, 672, 380, False)
  # segment.init_network(False)

  # segment.close()
  print("test")

if __name__ == "__main__":
    tf.app.run()
