from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import time
import os

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


class Segment:

  max_iterations = int(1e5 + 1)
  max_epochs = 100
  num_of_classes = 255
  image_resize = False
  image_width = 672
  image_height = 380
  keep_probability = None
  image = None
  annotation = None
  pred_annotation = None
  logits = None
  loss = None
  trainable_var = None
  train_op = None
  summary_op = None
  val_loss_sum_op = None
  train_records = None
  valid_records = None
  sess = None
  train_dataset_reader = None
  validation_dataset_reader = None
  saver = None
  summary_writer = None
  FLAGS = tf.flags.FLAGS

  def __init__(self, resize=False, width=672, height=380):
    self.image_resize = resize
    self.image_width = width
    self.image_height = height

  def vgg_net(self, weights, image):
    layers = (
      'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

      'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

      'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
      'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

      'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
      'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
      'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
      kind = name[:4]
      if kind == 'conv':
        kernels, bias = weights[i][0][0][0][0]
        # matconvnet: weights are [width, height, in_channels, out_channels]
        # tensorflow: weights are [height, width, in_channels, out_channels]
        kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
        bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
        current = utils.conv2d_basic(current, kernels, bias)
      elif kind == 'relu':
        current = tf.nn.relu(current, name=name)
        if self.FLAGS.debug:
          utils.add_activation_summary(current)
      elif kind == 'pool':
        current = utils.avg_pool_2x2(current)
      net[name] = current

    return net

  def inference(self, image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(self.FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
      image_net = self.vgg_net(weights, processed_image)
      conv_final_layer = image_net["conv5_3"]

      pool5 = utils.max_pool_2x2(conv_final_layer)

      W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
      b6 = utils.bias_variable([4096], name="b6")
      conv6 = utils.conv2d_basic(pool5, W6, b6)
      relu6 = tf.nn.relu(conv6, name="relu6")
      if self.FLAGS.debug:
          utils.add_activation_summary(relu6)
      relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

      W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
      b7 = utils.bias_variable([4096], name="b7")
      conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
      relu7 = tf.nn.relu(conv7, name="relu7")
      if self.FLAGS.debug:
          utils.add_activation_summary(relu7)
      relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

      W8 = utils.weight_variable([1, 1, 4096, self.num_of_classes], name="W8")
      b8 = utils.bias_variable([self.num_of_classes], name="b8")
      conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
      # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

      # now to upscale to actual image size
      deconv_shape1 = image_net["pool4"].get_shape()
      W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self.num_of_classes], name="W_t1")
      b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
      conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
      fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

      deconv_shape2 = image_net["pool3"].get_shape()
      W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
      b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
      conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
      fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

      shape = tf.shape(image)
      deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], self.num_of_classes])
      W_t3 = utils.weight_variable([16, 16, self.num_of_classes, deconv_shape2[3].value], name="W_t3")
      b_t3 = utils.bias_variable([self.num_of_classes], name="b_t3")
      conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

      annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

  def train(self, loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if self.FLAGS.debug:
      # print(len(var_list))
      for grad, var in grads:
        utils.add_gradient_summary(grad, var)

    return optimizer.apply_gradients(grads)

  def init_network(self, random_filenames):
    self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    self.image = tf.placeholder(tf.float32,
               shape=[None, self.image_height, self.image_width, 3], name="input_image")
    self.annotation = tf.placeholder(tf.int32,
                shape=[None, self.image_height, self.image_width, 1], name="annotation")

    self.pred_annotation, self.logits = self.inference(self.image, self.keep_probability)
    tf.summary.image("input_image", self.image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(self.annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(self.pred_annotation, tf.uint8), max_outputs=2)
    self.loss = tf.reduce_mean((
      tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,
                                                     tf.squeeze(self.annotation, squeeze_dims=[3]),
                                                     name="entropy")))
    tf.summary.scalar("training_loss", self.loss)

    self.trainable_var = tf.trainable_variables()
    if self.FLAGS.debug:
        for var in self.trainable_var:
            utils.add_to_regularization_and_summary(var)
        self.train_op = self.train(self.loss, self.trainable_var)

    print("Setting up summary op...")
    self.summary_op = tf.summary.merge_all()

    self.val_loss_sum_op = tf.summary.scalar("validation_loss", self.loss)

    print("Setting up image reader...")
    self.train_records, self.valid_records = \
      scene_parsing.read_dataset(self.FLAGS.data_dir, random_filenames)
    print(len(self.train_records))
    print(len(self.valid_records))
    self.max_iterations = self.max_epochs * len(self.train_records)

    print("Setting up dataset reader")
    image_options = {'resize': self.image_resize,
                     'image_height': self.image_height,
                     'image_width': self.image_width}
    if self.FLAGS.mode == 'train':
        self.train_dataset_reader = dataset.BatchDatset(
          self.train_records, self.FLAGS.batch_size, image_options)
        self.train_dataset_reader.start()
        # Wait for first images to load
        self.train_dataset_reader.wait_for_images()

    self.validation_dataset_reader = dataset.BatchDatset(
      self.valid_records, self.FLAGS.batch_size, image_options)
    self.validation_dataset_reader.start()
    # Wait for first images to load
    self.validation_dataset_reader.wait_for_images()

    self.sess = tf.Session()

    print("Setting up Saver...")
    self.saver = tf.train.Saver()
    self.summary_writer =  tf.summary.FileWriter(self.FLAGS.logs_dir, self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(self.FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      print("Model restored...")

  def train_network(self):

    for itr in xrange(self.max_iterations):
      train_images, train_annotations, train_image_names = \
        self.train_dataset_reader.next_batch(True)
      feed_dict = {self.image: train_images,
                   self.annotation: train_annotations,
                   self.keep_probability: 0.85}

      self.sess.run(self.train_op, feed_dict=feed_dict)

      if itr % 10 == 0:
        train_loss, summary_str = \
          self.sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        print("Step: %d, Train_loss:%g" % (itr, train_loss))
        self.summary_writer.add_summary(summary_str, itr)

      if itr % 500 == 0:
        valid_images, valid_annotations, val_image_names = \
          self.validation_dataset_reader.next_batch(True)
        valid_loss, val_summary_str = self.sess.run([self.loss, self.val_loss_sum_op],
              feed_dict={self.image: valid_images, self.annotation: valid_annotations,
                         self.keep_probability: 1.0})
        self.summary_writer.add_summary(val_summary_str, itr)
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
        #self.saver.save(self.sess, self.FLAGS.logs_dir + "model.ckpt", itr)

  def visualize_batch(self, data_reader, random_images, save_dir):
    valid_images, valid_annotations, valid_filenames = \
      data_reader.next_batch(random_images)
    pred = self.sess.run(self.pred_annotation,
                         feed_dict={self.image: valid_images,
                                    self.annotation: valid_annotations,
                                    self.keep_probability: 1.0})
    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)

    itr = 0
    for img_name in valid_filenames:
      utils.save_image(valid_images[itr].astype(np.uint8),
                       self.FLAGS.logs_dir, name=os.path.join(save_dir,
                                                              img_name['filename']+"_input"))
      utils.save_image(valid_annotations[itr].astype(np.uint8),
                       self.FLAGS.logs_dir, name=os.path.join(save_dir,
                                                              img_name['filename']+"_mask"))
      utils.save_image(pred[itr].astype(np.uint8),
                       self.FLAGS.logs_dir, name=os.path.join(save_dir,
                                                              img_name['filename']+"_pred"))
      itr += 1
      print("Saved image: ", img_name)

  def visualize_directory(self, data_reader, random_images, save_dir):
    total_count = int(len(data_reader.files) / self.FLAGS.batch_size)

    for idx in range(total_count):
      self.visualize_batch(data_reader, random_images, save_dir)