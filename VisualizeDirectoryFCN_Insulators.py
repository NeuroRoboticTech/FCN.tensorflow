import numpy as np
import FCN
import tensorflow as tf
import FCN_env_vars as EnvVars

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", EnvVars.logs_dir, "path to logs directory")
tf.flags.DEFINE_string("data_dir", EnvVars.data_dir + "\\Data_Zoo\\Insulators", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", EnvVars.data_dir + "\\Model_zoo\\", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")


def main(argv=None):
  segment = FCN.Segment(False, 700, 700, 3, -1, False)
  segment.init_network(False)

  #segment.visualize_directory(
  #  segment.train_dataset_reader, False, False,
  #  EnvVars.data_dir + "/weeds/errors/training",
  #  EnvVars.data_dir + "/Weed_errors/errors/training")

  segment.visualize_directory(
     segment.validation_dataset_reader, False, False,
     EnvVars.data_dir + "\\Data_Zoo\\Insulators\\data\\images\\validation")

  segment.close()

if __name__ == "__main__":
    tf.app.run()
