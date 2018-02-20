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
import psycopg2
import scipy.misc as misc

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


class Segment:

  max_iterations = int(1e5 + 1)
  max_epochs = 10
  num_of_classes = 255
  image_resize = False
  image_width = 672
  image_height = 380
  image_channels = 3

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

  train_accuracy = None
  val_accuracy = None

  db_logging = True
  conn = None
  cur = None
  run_name = "run1"
  run_descr = "test"
  run_id = 0
  force_size_idx = -1

  def __init__(self, resize=False, width=672, height=380, image_channels=3, force_size_idx = -1, db_logging=True):
    self.image_resize = resize
    self.image_width = width
    self.image_height = height
    self.image_channels = image_channels
    self.force_size_idx = force_size_idx
    self.db_logging = db_logging

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

        if self.image_channels == 1 and kernels.shape[2] == 3:
          # If we are using a single channel the remove the other two
          # channels from the weights.
          kernels = kernels[:, :, 0, :].reshape((3, 3, 1, 64))

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
    if self.image_channels == 1:
      mean = mean[:, :, [0]]

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
      deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.num_of_classes])
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
    # Create DB entries.
    if self.db_logging:
      self.conn = psycopg2.connect("dbname=FCN_Data user=postgres password=abc123")
      self.cur = self.conn.cursor()
      self.cur.execute("INSERT INTO experiment (name, description) VALUES (%s, %s)",
                     (self.run_name, self.run_descr))
      self.cur.execute("SELECT MAX(id) FROM experiment;")
      r = self.cur.fetchone()
      self.run_id = int(r[0])
      self.conn.commit()

    # create newtork
    self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    self.image = tf.placeholder(tf.float32,
               shape=[None, self.image_height, self.image_width, self.image_channels], name="input_image")
    self.annotation = tf.placeholder(tf.int32,
                shape=[None, self.image_height, self.image_width, 1], name="annotation")

    self.pred_annotation, self.logits = self.inference(self.image, self.keep_probability)
    tf.summary.image("input_image", self.image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(self.annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(self.pred_annotation, tf.uint8), max_outputs=2)
    self.loss = tf.reduce_mean((
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                     labels=tf.squeeze(self.annotation, squeeze_dims=[3]),
                                                     name="entropy")))


    tf.summary.scalar("training_loss", self.loss)

    # Define the train/val accuracy variables.
    # self.train_accuracy = tf.Variable(0)
    # self.val_accuracy = tf.Variable(0)

    # tf.summary.scalar("training_accuracy", self.train_accuracy)
    # tf.summary.scalar("validation_accuracy", self.val_accuracy)

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
    self.max_iterations = self.max_epochs * len(self.train_records) + 1

    print("Setting up dataset reader")
    image_options = {'resize': self.image_resize,
                     'image_height': self.image_height,
                     'image_width': self.image_width,
                     'image_channels': self.image_channels}
    allowed_mask_vals = [0, 128, 192]
    if self.FLAGS.mode == 'train' or self.FLAGS.mode == 'visualize':
        self.train_dataset_reader = dataset.BatchDatset(
          self.train_records, self.FLAGS.batch_size, allowed_mask_vals, image_options)
        self.train_dataset_reader.start()
        # Wait for first images to load
        self.train_dataset_reader.wait_for_images()

    self.validation_dataset_reader = dataset.BatchDatset(
      self.valid_records, self.FLAGS.batch_size, allowed_mask_vals, image_options)
    self.validation_dataset_reader.start()
    # Wait for first images to load
    self.validation_dataset_reader.wait_for_images()

    self.sess = tf.Session()

    print("Setting up Saver...")
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    self.summary_writer = tf.summary.FileWriter(self.FLAGS.logs_dir, self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

    print("Restoring model.")
    ckpt = tf.train.get_checkpoint_state(self.FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      print("Model restored...")

  def train_network(self):

    itr = 0
    for epoch in xrange(self.max_epochs):
      for epoch_itr in xrange(len(self.train_records)):
        train_images, train_annotations, train_image_names = \
          self.train_dataset_reader.next_batch(True, False, self.force_size_idx)
        feed_dict = {self.image: train_images,
                     self.annotation: train_annotations,
                     self.keep_probability: 0.85}

        self.sess.run(self.train_op, feed_dict=feed_dict)

        if itr % 10 == 0:
          train_loss, summary_str = \
            self.sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
          print("Step: %d, Train_loss:%g, file: %s" % (itr, train_loss,
                                                       self.train_dataset_reader.filename['filename']))
          self.summary_writer.add_summary(summary_str, itr)
          self.save_loss_to_db(epoch, itr, train_loss, self.train_dataset_reader, True)

        if itr % 500 == 0:
          valid_images, valid_annotations, val_image_names = \
            self.validation_dataset_reader.next_batch(True, True, self.force_size_idx)
          valid_loss, val_summary_str = self.sess.run([self.loss, self.val_loss_sum_op],
                feed_dict={self.image: valid_images, self.annotation: valid_annotations,
                           self.keep_probability: 1.0})
          self.summary_writer.add_summary(val_summary_str, itr)
          print("%s ---> Validation_loss: %g, file: %s" % (datetime.datetime.now(), valid_loss,
                                                       self.validation_dataset_reader.filename['filename']))
          self.save_loss_to_db(epoch, itr, valid_loss, self.validation_dataset_reader, False)
          self.saver.save(self.sess, self.FLAGS.logs_dir + "model.ckpt", itr)

        itr += 1

      # Calculate accuracies for all validation images.
      train_accuracy_val = self.calc_accuracies_for_images(epoch, self.train_dataset_reader, True, True)
      val_accuracy_val = self.calc_accuracies_for_images(epoch, self.validation_dataset_reader, True, False)

      # Not set our tf vars for accuracy and update them.
      # train_accuracy_assign_op = self.train_accuracy.assign(train_accuracy_val)
      # val_accuracy_assign_op = self.val_accuracy.assign(val_accuracy_val)
      # self.sess.run([train_accuracy_assign_op, val_accuracy_assign_op])

      print("Training Accuracy: %g", train_accuracy_val)
      print("Validation Accuracy: %g", val_accuracy_val)
      print("****************** Epochs completed: " + str(epoch) + "******************")

  def save_visualized_batch_images(self, valid_images, valid_annotations,
    valid_filenames, pred, total_accuracy, mask_errors, save_dir):

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
      utils.save_image(mask_errors[itr].astype(np.uint8),
                       self.FLAGS.logs_dir, name=os.path.join(save_dir,
                                                              img_name['filename']+"_errors"))
      itr += 1
      print("Saved image: ", img_name)

  def visualize_batch(self, data_reader, random_images, train_record, save_dir):
    valid_images, valid_annotations, \
    valid_filenames, pred, total_accuracy, mask_errors = \
      self.calc_accuracy_for_batch(0, data_reader, random_images, train_record)
    self.save_visualized_batch_images(valid_images, valid_annotations, valid_filenames,
                                pred, total_accuracy, mask_errors, save_dir)

  def visualize_directory(self, data_reader, random_images, train_record, save_dir):
    total_count = int(len(data_reader.files) / self.FLAGS.batch_size)

    for idx in range(total_count):
      self.visualize_batch(data_reader, random_images, train_record, save_dir)

  def visualize_error_directory(self, data_reader, data_list, train_record,
                                save_out, save_dir):

    for data_set in data_list:
      valid_images, valid_annotations, \
      valid_filenames, pred, total_accuracy, mask_errors = \
        self.calc_accuracy_for_data_set(0, data_reader, data_set,
                                        train_record, save_out)
      self.save_visualized_batch_images(valid_images, valid_annotations, valid_filenames,
                                  pred, total_accuracy, mask_errors, save_dir)

  def close(self):
    if self.train_dataset_reader is not None:
      self.train_dataset_reader.exit_thread = True

    if self.validation_dataset_reader is not None:
      self.validation_dataset_reader.exit_thread = True

    if self.db_logging:
      self.cur.close()
      self.conn.close()

  def save_loss_to_db(self, epoch, itr, loss, dataset_reader, train_record):
    if self.db_logging:
      self.cur.execute("INSERT INTO losses (experiment_id, epoch, "
                     "iteration, loss, training, image, flip, "
                     "rotation, size_idx, cut_x, cut_y) VALUES "
                     "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                     (self.run_id, epoch, itr, float(loss), train_record,
                      dataset_reader.filename['filename'], dataset_reader.flip,
                      dataset_reader.rotation, dataset_reader.size_idx,
                      dataset_reader.cut_x, dataset_reader.cut_y))
      self.conn.commit()

  def calc_accuracy_for_image(self, mask_orig, mask_pred):
    # Find the difference between all pixel values.
    # When pixels match the diff will be 0. When they do not they will be some
    # other value.
    diff_mask = np.absolute(mask_orig - mask_pred)

    # Generate an array with 1  for
    errors = np.where(diff_mask > 2)
    pixels_incorrect = len(errors[0])
    accuracy = float(mask_orig.size - pixels_incorrect)/float(mask_orig.size) * 100.0

    mask_errors = np.zeros(mask_orig.shape, np.uint8)
    errors_height = errors[0].tolist()
    errors_width = errors[1].tolist()
    mask_errors[errors_height, errors_width] = 255

    misc.imsave('final_errors.png', mask_errors)

    return accuracy, mask_errors

  def save_accuracy_to_db(self, epoch, itr, img_name,
                          accuracy, train_record, data_reader):
    if self.db_logging:
      self.cur.execute("INSERT INTO accuracies (experiment_id, epoch, "
                     "iteration, accuracy, training, image, flip, "
                     "rotation, size_idx, cut_x, cut_y) VALUES "
                     "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                     (self.run_id, epoch, itr, float(accuracy), train_record,
                      data_reader.filename['filename'], data_reader.flip,
                      data_reader.rotation, data_reader.size_idx,
                      data_reader.cut_x, data_reader.cut_y))
      self.conn.commit()

  def save_avg_accuracy_to_db(self, epoch, avg_accuracy, train_record):
    if self.db_logging:
      self.cur.execute("INSERT INTO average_accuracies (experiment_id, epoch, "
                     "accuracy, training) VALUES "
                     "(%s, %s, %s, %s)",
                     (self.run_id, epoch, float(avg_accuracy), train_record))
      self.conn.commit()

  def calc_accuracy_for_batch_images(self, epoch, data_reader, train_record,
                                     valid_images, valid_annotations, valid_filenames,
                                     pred):
    itr = 0
    total_accuracy = 0
    mask_errors = []
    for img_name in valid_filenames:
      # Using int16 here so I can do differences of two images.
      mask_orig = valid_annotations[itr].astype(np.int16)
      mask_pred = pred[itr].astype(np.int16)
      accuracy, mask_error = self.calc_accuracy_for_image(mask_orig, mask_pred)
      total_accuracy += accuracy
      itr += 1
      mask_errors.append(mask_error)
      self.save_accuracy_to_db(epoch, itr, img_name['filename'],
                               accuracy, train_record, data_reader)

    return total_accuracy, mask_errors

  def calc_accuracy_for_data_set(self, epoch, data_reader, data_set, train_record, save_out):
    valid_images, valid_annotations, valid_filenames = \
      data_reader.next_batch_from_list(data_set[0], data_set[1], data_set[2],
                                       data_set[3], data_set[4], data_set[5],
                                       save_out)
    pred = self.sess.run(self.pred_annotation,
                         feed_dict={self.image: valid_images,
                                    self.annotation: valid_annotations,
                                    self.keep_probability: 1.0})
    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)

    total_accuracy, mask_errors = \
      self.calc_accuracy_for_batch_images(epoch, data_reader, train_record,
                                          valid_images, valid_annotations,
                                          valid_filenames, pred)
    return valid_images, valid_annotations, \
           valid_filenames, pred, total_accuracy, mask_errors

  def calc_accuracy_for_batch(self, epoch, data_reader, random_images, train_record):
    valid_images, valid_annotations, valid_filenames = \
      data_reader.next_batch(random_images, False, self.force_size_idx)
    pred = self.sess.run(self.pred_annotation,
                         feed_dict={self.image: valid_images,
                                    self.annotation: valid_annotations,
                                    self.keep_probability: 1.0})
    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)

    total_accuracy, mask_errors = \
      self.calc_accuracy_for_batch_images(epoch, data_reader, train_record,
                                          valid_images, valid_annotations,
                                          valid_filenames, pred)
    return valid_images, valid_annotations, \
           valid_filenames, pred, total_accuracy, mask_errors

  def calc_accuracies_for_images(self, epoch, data_reader, random_images, train_record):

    total_count = int(len(data_reader.files) / self.FLAGS.batch_size)

    total_accuracy = 0.0
    for idx in range(total_count):
      valid_images, valid_annotations, \
      valid_filenames, pred, accuracy, mask_errors = \
        self.calc_accuracy_for_batch(epoch, data_reader, random_images, train_record)
      total_accuracy += accuracy

    avg_accuracy = float(total_accuracy / total_count)
    self.save_avg_accuracy_to_db(epoch, avg_accuracy, train_record)

    return avg_accuracy

  def generate_mask_for_unlabeled_image(self, img):

    channels = 1
    if len(img.shape) > 2:
        channels = img.shape[2]

    img_batch = np.array(img).reshape(
      (1, img.shape[0], img.shape[1], channels))

    blank_mask = np.zeros((1, img.shape[0], img.shape[1], 1))

    pred = self.sess.run(self.pred_annotation,
                         feed_dict={self.image: img_batch,
                                    self.annotation: blank_mask,
                                    self.keep_probability: 1.0})
    pred = np.squeeze(pred, axis=3)
    pred = np.squeeze(pred, axis=0)
    # pred = np.reshape(pred.shape[1], pred.shape[2])
    return pred
