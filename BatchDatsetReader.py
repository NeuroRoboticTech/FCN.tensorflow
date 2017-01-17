"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import cv2
import threading
import time
import random

def rotate_img(img, angle, center=None, scale=1.0):
  # grab the dimensions of the image
  (h, w) = img.shape[:2]

  # if the center is None, initialize it as the center of
  # the image
  if center is None:
    center = (w / 2, h / 2)

  # perform the rotation
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(img, M, (w, h))

  # return the rotated image
  return rotated


class BatchDatset (threading.Thread):
    files = []
    images = []
    image_files = []
    annotations = []
    image_options = {}
    batch_offset = 0
    batch_size = 0
    start_idx = 0
    end_idx = 0
    epochs_completed = 0
    final_height = 0
    final_width = 0
    load_next_images = True
    exit_thread = False
    lock = threading.Lock()

    def __init__(self, records_list, batch_size, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)

        threading.Thread.__init__(self)

        self.files = records_list
        self.image_options = image_options
        self.batch_size = batch_size
        self.start_idx = 0
        self.batch_offset = self.batch_size
        self.end_idx = self.batch_offset
        self.final_height = int(self.image_options["image_height"])
        self.final_width = int(self.image_options["image_width"])

    def _read_images(self):
        self.__channels = True
        self.image_files = self.files[self.start_idx:self.end_idx]
        self.images = np.array([self._transform(filename['image'])
                                for filename in self.files[self.start_idx:self.end_idx]])
        self.__channels = False
        self.annotations = np.array([self._transform(filename['annotation'])
                                     for filename in self.files[self.start_idx:self.end_idx]])
        # print (self.images.shape)
        # print (self.annotations.shape)

    def _transform(self, filename):
        image = misc.imread(filename)
        # print("Read image: ", filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_height = int(self.image_options["image_height"])
            resize_width = int(self.image_options["image_width"])
            resize_image = misc.imresize(image,
                                         [resize_height, resize_width], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def _random_transform(self, img, annot):
      # Flip the image around the vertical axis randomly
      if np.random.randint(0, 100) > 50:
        new_img = cv2.flip(img, 1)
        new_annot = cv2.flip(annot, 1)
      else:
        new_img = img
        new_annot = annot

      # Rotate the image if needed. Use a normal distribution
      # and truncate it to an integer to get the rotation degrees
      # to use.
      rotate_deg = int(np.random.normal(0, 8))
      if rotate_deg != 0:
        new_img = rotate_img(new_img, rotate_deg)
        new_annot = rotate_img(new_annot, rotate_deg)

      # Find out how many multiples the final image is compared
      # to the input image.
      img_width_multiple = int(img.shape[1] / self.final_width)

      # Randomly choose a size to use
      size_idx = np.random.randint(3, img_width_multiple)
      cut_width = int(size_idx * self.final_width)
      cut_height = int(cut_width * (self.final_height/self.final_width))

      # Randomly choose the pixel for the top-left corner where
      # we will begin the cut.
      area_width = img.shape[1] - cut_width
      area_height = img.shape[0] - cut_height
      cut_x = np.random.randint(0, area_width-1)
      cut_y = np.random.randint(0, area_height-1)

      # Cut out a section of the large image to resize to the
      # smaller section.
      cut_img = new_img[cut_y:(cut_y+cut_height), cut_x:(cut_x+cut_width)]
      cut_annot = new_annot[cut_y:(cut_y+cut_height), cut_x:(cut_x+cut_width)]

      #cv2.imwrite('F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/cut.jpg', cut_img)

      # Randomly alter the contrast and brightness of the cut image
      # use normal distribution around 1 for contrast multiplier
      contrast = np.random.normal(1.0, 0.02)
      brightness = np.random.randint(-10, 10)

      cut_img = (contrast * cut_img) + brightness

      #cv2.imwrite('F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/cut_bright.jpg', cut_img)

      # Last, resize the cut images to match the final image size.
      final_img = cv2.resize(cut_img, (self.final_width, self.final_height), interpolation=cv2.INTER_AREA)
      final_annot = cut_annot[::size_idx, ::size_idx]

      cv2.imwrite('F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/final_img.jpg', final_img)
      cv2.imwrite('F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/final_mask.jpg', final_annot)

      return final_img, final_annot

    def run(self):
      print("Starting BatchDataSet Thread")
      self._load_images_thread()
      print("Exiting BatchDataSet Thread")

    def _load_images_thread(self):

      while not self.exit_thread:
        if self.load_next_images:
          with self.lock:
            self._read_images()
            self.load_next_images = False

        else:
          time.sleep(0.002)

    def next_batch_old(self):
      start = self.batch_offset
      self.batch_offset += self.batch_size
      if self.batch_offset > self.images.shape[0]:
        # Finished epoch
        self.epochs_completed += 1
        print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
        # Shuffle the data
        perm = np.arange(self.images.shape[0])
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.annotations = self.annotations[perm]
        # Start next epoch
        start = 0
        self.batch_offset = self.batch_size

      end = self.batch_offset
      return self.images[start:end], self.annotations[start:end]

    def wait_for_images(self):
      while len(self.images) < self.batch_size:
        while self.load_next_images:
          time.sleep(0.002)

        if len(self.images) < self.batch_size:
          self.load_next_images = True

    def next_batch(self, random_mod):

      # Wait for the images to be loaded if they are not already in place
      self.wait_for_images()

      with self.lock:
        self.start_idx = self.batch_offset
        self.batch_offset += self.batch_size
        if self.batch_offset > len(self.files):
          # Finished epoch
          self.epochs_completed += 1
          print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
          # Shuffle the data
          random.shuffle(self.files)
          # Start next epoch
          self.start_idx = 0
          self.batch_offset = self.batch_size

        self.end_idx = self.batch_offset

        img_batch_list = []
        annot_batch_list = []

        idx = 0
        for img in self.images:
          annot = self.annotations[idx]

          if random_mod:
            img_trans, annot_trans = self._random_transform(img, annot)
          else:
            img_trans = img
            annot_trans = annot

          img_batch_list.append(img_trans)
          annot_batch_list.append(annot_trans)
          idx += 1

        img_batch = np.array(img_batch_list)
        annot_batch = np.array(annot_batch_list).reshape(
          (len(annot_batch_list), self.final_height, self.final_width, 1))

        # now kick off thread to load next batch of images.
        # print("processed next batch.")
        self.load_next_images = True

        return img_batch, annot_batch, self.image_files

    def get_random_batch(self, batch_size):
      indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
      return self.images[indexes], self.annotations[indexes]
