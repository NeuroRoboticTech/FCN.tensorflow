import numpy as np
import FCN
import tensorflow as tf
import FCN_env_vars as EnvVars
import scipy.misc as misc

project_Dir = "Humans"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", EnvVars.logs_dir, "path to logs directory")
tf.flags.DEFINE_string("data_dir", EnvVars.data_dir + "/Data_Zoo/" + project_Dir, "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", EnvVars.data_dir + "/model_zoo", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

image_width = 224
image_height = 224
image_channels = 3
force_size_idx = 1


def main(argv=None):
    segment = FCN.Segment(True, image_width, image_height, 
                          image_channels, force_size_idx, False)
    segment.init_network(False)

    img = misc.imread('/media/ubuntu/SDRoot/FCN_Tensorflow/data/Data_Zoo/Humans/data/images/training/2007_000170.jpg')
    scaled_img = misc.imresize(img, (image_height, image_width))
    mask = segment.generate_mask_for_unlabeled_image(scaled_img)

    segment.close()

    misc.imsave('/media/ubuntu/SDRoot/FCN_Tensorflow/data/Data_Zoo/Humans/2007_000170_mask.png', mask)

if __name__ == "__main__":
    tf.app.run()
