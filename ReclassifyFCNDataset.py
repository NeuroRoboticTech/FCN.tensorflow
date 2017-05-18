import numpy as np
import FCN
import tensorflow as tf
import FCN_env_vars as EnvVars
import scipy.misc as misc

project_Dir = "Weeds"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", EnvVars.logs_dir, "path to logs directory")
tf.flags.DEFINE_string("data_dir", EnvVars.data_dir + "/Data_Zoo/" + project_Dir, "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", EnvVars.data_dir + "/model_zoo", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

image_width = 672
image_height = 380


def main(argv=None):
    segment = FCN.Segment(True, 672, 380, 3, -1, False)
    segment.init_network(False)

    img = misc.imread('D:/Projects/FCN_Tensorflow/data/Data_Zoo/Weeds/data/images/training/P1090526.JPG')
    scaled_img = misc.imresize(img, (image_height, image_width))
    mask = segment.generate_mask_for_unlabeled_image(scaled_img)

    segment.close()

    misc.imsave('D:/Projects/FCN_Tensorflow/data/Data_Zoo/Weeds/P1090526_mask.PNG', mask)

if __name__ == "__main__":
    tf.app.run()
