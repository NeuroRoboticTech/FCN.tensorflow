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

train_data = [['P1090996', True, 11, 3, 416, 1500],
              ['P1090996', True, 4, 5, 286, 612],
              ['P1040709', False, -6, 4, 555, 1458],
              ['P1090999', False, 14, 6, 0, 112],
              ['P1040692', True, -2, 4, 738, 1117],
              ['P1100002', True, -2, 3, 392, 969],
              ['P1040666', False, 0, 5, 366, 98],
              ['P1040767', True, 10, 3, 1240, 680],
              ['P1070820', True, 5, 6, 0, 411],
              ['P1020756', False, -5, 3, 475, 352],
              ['P1100004', False, -6, 6, 0, 137],
              ['P1050344', True, 0, 3, 1340, 33],
              ['P1070163', False, -13, 3, 418, 134],
              ['P1040599', False, -5, 6, 0, 138],
              ['P1080254', True, 10, 3, 1911, 9],
              ['P1060221', False, -18, 3, 1212, 511],
              ['P1100007', True, 10, 3, 993, 166],
              ['P1080203', True, 5, 6, 0, 389],
              ['P1090945', False, 0, 3, 636, 911],
              ['P1070156', False, 0, 4, 410, 70],
              ['P1070134', False, 0, 3, 1447, 242],
              ['P1070394', True, 8, 6, 0, 101],
              ['P1070384', False, 4, 5, 124, 558],
              ['P1070109', True, -4, 4, 1073, 640],
              ['P1070386', True, 1, 4, 761, 1170],
              ['P1070133', False, 1, 6, 0, 621],
              ['P1080258', True, -3, 3, 1770, 775],
              ['P1080271', False, -15, 5, 263, 135],
              ['P1100012', True, -11, 6, 0, 556],
              ['P1080163', True, -1, 4, 609, 823],
              ['P1070082', False, -1, 5, 209, 751],
              ['P1070174', True, -1, 3, 1713, 723],
              ['P1070389', True, 7, 5, 55, 839],
              ['P1100006', True, 1, 3, 733, 10],
              ['P1040705', True, 0, 6, 0, 676],
              ['P1070393', False, 16, 3, 973, 1140],
              ['P1070393', True, 10, 6, 0, 500],
              ['P1070127', False, 7, 4, 225, 1311],
              ['P1080275', False, 25, 3, 1858, 1072],
              ['P1040551', True, 9, 3, 1504, 951],
              ['P1070413', True, -1, 3, 711, 1299],
              ['P1100013', False, 8, 5, 429, 274],
              ['P1090980', True, 9, 4, 1094, 222],
              ['P1070176', False, 7, 3, 580, 1851]
              ]

val_data = [['P1070113', False, -7, 6, 0, 354],
            ['P1040693', False, -10, 5, 224, 232],
            ['P1080253', False, 7, 4, 1187, 111],
            ['P1070342', True, -12, 6, 0, 79],
            ['P1090998', True, -2, 3, 674, 214],
            ['P1070129', False, 11, 5, 29, 657],
            ['P1070189', True, 4, 6, 0, 535],
            ['P1100008', False, 8, 4, 662, 219]
            ]


def main(argv=None):
  segment = FCN.Segment()
  segment.init_network(False)

  segment.visualize_error_directory(
    segment.validation_dataset_reader, val_data, True, True,
    "F:/Projects/FCN_tensorflow/data/Data_zoo/Weed_Errors/errors/validation")
  segment.close()

if __name__ == "__main__":
    tf.app.run()

