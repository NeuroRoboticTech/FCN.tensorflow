"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import threading
import time
import random
import sys
import cv2

def rotate_img(img, angle, center=None, scale=1.0):
  # grab the dimensions of the image
  (h, w) = img.shape[:2]

  # if the center is None, initialize it as the center of
  # the image
  if center is None:
    center = (w / 2, h / 2)

  # perform the rotation
  if scale != 1.0:
    scaled = misc.imresize(img, scale)
  else:
    scaled = img

  rotated = misc.imrotate(scaled, angle)

  # misc.imsave('D:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/before_rotate_img.jpg', img)
  # misc.imsave('D:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/after_rotate_img.jpg', rotated)

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

    filename = ""
    flip = False
    rotation = 0.0
    size_idx = 0
    cut_x = 0
    cut_y = 0

    def __init__(self, records_list, batch_size, allowed_mask_vals, image_options={}):
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
        self.__channels = True  # Default to true
        self.allowed_mask_vals = allowed_mask_vals

        self.filename = ""
        self.flip = False
        self.rotation = 0
        self.size_idx = 0
        self.cut_x = 0
        self.cut_y = 0

    def _read_images(self):
        self.image_files = self.files[self.start_idx:self.end_idx]
        self._read_images_files()

    def _read_set_image(self, img_filename):
        self.image_files = [img for img in self.files if img['filename'] == img_filename]
        self._read_images_files()

    def _read_images_files(self):
      self.__channels = True
      images = np.array([self.transform(filename['image'])
                              for filename in self.image_files])
      self.__channels = False
      annotations = np.array([self.transform(filename['annotation'])
                                   for filename in self.image_files])

      self.images.clear()
      self.annotations.clear()

      for idx in range(0, len(images)):
        img_trans, annot_trans =  self.random_transform(images[idx], annotations[idx])
        self.images.append(img_trans)
        self.annotations.append(annot_trans)

      # print (self.images.shape)
      # print (self.annotations.shape)

    def transform(self, filename):
        if self.__channels and self.image_options['image_channels'] == 1:
          greyscale = True
        else:
          greyscale = False

        image = misc.imread(filename, flatten=greyscale)
        # print("Read image: ", filename)
        if self.__channels and not greyscale and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_height = int(self.image_options["image_height"])
            resize_width = int(self.image_options["image_width"])
            resize_image = misc.imresize(image,
                                         [resize_height, resize_width], interp='nearest')
        else:
            resize_image = image

        if greyscale:
          resize_image = resize_image.reshape((resize_height, resize_width, 1))

        #image_ratio = resize_image.shape[1] / resize_image.shape[0]
        #if image_ratio != self.final_ratio:
        #    print("Ratio wrong")

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset


    def random_transform(self, img, annot, save_out=False, force_size_idx=-1, force_flip=-1, force_rot=-1000, force_cut_x=-1, force_cut_y=-1):
      if force_flip == -1:
        if np.random.randint(0, 100) > 50:
          flip = True
        else:
          flip = False
      else:
        if force_flip == 0:
          flip = False
        else:
          flip = True

      if force_rot == -1000:
        rotate_deg = int(np.random.normal(0, 8))
      else:
        rotate_deg = force_rot

      if force_size_idx > 0:
        size_idx = force_size_idx
      else:
        # Find out how many multiples the final image is compared
        # to the input image.
        img_width_multiple = int(img.shape[1] / self.final_width)
        # Randomly choose a size to use
        if(img_width_multiple + 2 > 3):
          size_idx = np.random.randint(3, img_width_multiple + 2)
        else:
          size_idx = 1

        if size_idx >= img_width_multiple:  # Give larger image a greater chance of being picked.
          size_idx = img_width_multiple

        if size_idx < 5:
          # Check to make sure that more than half of the image is not zeros
          non_zero_annot = np.where(annot != 0)
          non_zero_annot_count = len(non_zero_annot[0])
          non_zero_perc = float(non_zero_annot_count)/float(annot.size)
          if non_zero_perc > 0.5:
            size_idx = img_width_multiple

      final_img, final_annot = self.transform_img(img, annot, save_out, flip, rotate_deg, size_idx, force_cut_x, force_cut_y)
      return final_img, final_annot

    def transform_img(self, img, annot, save_out, flip, rotate_deg,
                          size_idx, cut_x_in, cut_y_in):

        # Flip the image around the vertical axis randomly
      if flip:
        new_img = np.fliplr(img)
        new_annot = np.fliplr(annot)

        # misc.imsave('D:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/before_img.jpg', img)
        # misc.imsave('D:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/flipped_img.jpg', new_img)
        # misc.imsave('D:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/before_mask.png', annot)
        # misc.imsave('D:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/flipped_mask.png', new_annot)

        self.flip = True
      else:
        new_img = img
        new_annot = annot
        self.flip = False

        # Rotate the image if needed. Use a normal distribution
      # and truncate it to an integer to get the rotation degrees
      # to use.
      self.rotation = rotate_deg
      if rotate_deg != 0:
        M = cv2.getRotationMatrix2D((new_annot.shape[1] / 2, new_annot.shape[0] / 2), rotate_deg, 1)
        new_annot2 = cv2.warpAffine(new_annot, M, (new_annot.shape[1], new_annot.shape[0]))

        new_img = rotate_img(new_img, rotate_deg)
        new_annot = rotate_img(new_annot, rotate_deg)


      cut_width = int(size_idx * self.final_width)
      cut_height = int(cut_width * (self.final_height/self.final_width))
      self.size_idx = size_idx

      # Randomly choose the pixel for the top-left corner where
      # we will begin the cut.
      area_width = img.shape[1] - cut_width
      area_height = img.shape[0] - cut_height
      if area_width > 1:
        cut_x = np.random.randint(0, area_width-1)
      else:
        cut_x = 0

      if area_height > 1:
        cut_y = np.random.randint(0, area_height - 1)
      else:
        cut_y = 0

      if cut_x_in >= 0:
        cut_x = cut_x_in

      if cut_y_in >= 0:
        cut_y = cut_y_in

      self.cut_x = cut_x
      self.cut_y = cut_y

      # Cut out a section of the large image to resize to the
      # smaller section.
      cut_img = new_img[cut_y:(cut_y+cut_height), cut_x:(cut_x+cut_width)]
      cut_annot = new_annot[cut_y:(cut_y+cut_height), cut_x:(cut_x+cut_width)]

      # if cut_img.shape[0] != 340:
      #   raise ValueError("Image height incorrect: " + str(cut_img.shape[0]))
      # if cut_img.shape[1] != 512:
      #   raise ValueError("Image width incorrect: " + str(cut_img.shape[1]))

      #cv2.imwrite('F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/cut.jpg', cut_img)

      # Randomly alter the contrast and brightness of the cut image
      # use normal distribution around 1 for contrast multiplier
      #contrast = 1 #np.random.normal(1.0, 0.02)
      #brightness = np.random.randint(-10, 10)

      #cut_img = (contrast * cut_img) + brightness

      #cv2.imwrite('F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/cut_bright.jpg', cut_img)

      # Last, resize the cut images to match the final image size.
      final_img = misc.imresize(cut_img, (self.final_height, self.final_width))
      # final_img = cv2.resize(cut_img, (self.final_width, self.final_height), interpolation=cv2.INTER_AREA)
      #final_annot = cv2.resize(cut_annot, (self.final_width, self.final_height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
      final_annot = misc.imresize(cut_annot, (self.final_height, self.final_width), interp='nearest')
      #final_annot = cut_annot[::size_idx, ::size_idx]

      if save_out:
        misc.imsave('final_orig.jpg', img)
        misc.imsave('final_img.jpg', final_img)
        misc.imsave('final_mask.png', cut_annot)

      final_annot = self.removeDisallowedMaskValues(final_annot)

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

    def wait_for_images(self):
      while len(self.images) < self.batch_size:
        while self.load_next_images:
          time.sleep(0.002)

        if len(self.images) < self.batch_size:
          self.load_next_images = True

    def next_batch(self, random_mod, save_out, force_size_idx=-1, force_flip=-1, force_rot=-1000, force_cut_x=-1, force_cut_y=-1):

      # Wait for the images to be loaded if they are not already in place
      self.wait_for_images()

      with self.lock:
        self.start_idx = self.batch_offset
        self.batch_offset += self.batch_size
        if self.batch_offset > len(self.files):
          # Finished epoch
          self.epochs_completed += 1
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
          self.filename = self.image_files[idx]

          img_batch_list.append(img)
          annot_batch_list.append(annot)
          idx += 1

        img_batch = np.array(img_batch_list).reshape(
            (len(img_batch_list), self.final_height, self.final_width, self.image_options['image_channels']))

        annot_batch = np.array(annot_batch_list).reshape(
          (len(annot_batch_list), self.final_height, self.final_width, 1))

        # now kick off thread to load next batch of images.
        # print("processed next batch.")
        self.load_next_images = True

        return img_batch, annot_batch, self.image_files

    def get_random_batch(self, batch_size):
      indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
      return self.images[indexes], self.annotations[indexes]

    def next_batch_from_list(self, img_file, flip, rotate_deg,
                             size_idx, cut_x_in, cut_y_in, save_out):

      self._read_set_image(img_file)

      img_batch_list = []
      annot_batch_list = []

      idx = 0
      for img in self.images:
        annot = self.annotations[idx]
        self.filename = self.image_files[idx]

        img_trans, annot_trans = self.transform_img(
                        img, annot, save_out, flip, rotate_deg,
                        size_idx, cut_x_in, cut_y_in)

        img_batch_list.append(img_trans)
        annot_batch_list.append(annot_trans)
        idx += 1

      img_batch = np.array(img_batch_list)
      annot_batch = np.array(annot_batch_list).reshape(
        (len(annot_batch_list), self.final_height, self.final_width, 1))

      return img_batch, annot_batch, self.image_files


    def removeDisallowedMaskValues(self, mask):

        #misc.imsave('new_mask_orig.png', mask)

        for idx in range(1, len(self.allowed_mask_vals)):
            low_val = self.allowed_mask_vals[idx-1]
            high_val = self.allowed_mask_vals[idx]
            half_val = int(low_val + ((self.allowed_mask_vals[idx] - self.allowed_mask_vals[idx-1])/2))

            lower_start_val = low_val + 1
            lower_end_val =  half_val

            upper_start_val = lower_end_val + 1
            upper_end_val = high_val -1

            low_range = cv2.inRange(mask, lower_start_val, lower_end_val)
            low_range_inv = cv2.bitwise_not(low_range)
            high_range = cv2.inRange(mask, upper_start_val, upper_end_val)
            high_range_inv = cv2.bitwise_not(high_range)

            mask = cv2.bitwise_and(mask, mask, mask = low_range_inv)
            mask = cv2.bitwise_and(mask, mask, mask = high_range_inv)

            mask_fill = ((low_range/255 * low_val) + (high_range/255 * high_val))
            mask = mask + mask_fill

            #misc.imsave('new_mask_out_' + str(idx) + '.png', mask_fill)
            #misc.imsave('new_mask_' + str(idx) + '.png', mask)

        # Now get the pixels between the last one and the 255 value
        low_val = self.allowed_mask_vals[-1]
        low_range = cv2.inRange(mask, self.allowed_mask_vals[-1]+1, 255)
        low_range_inv = cv2.bitwise_not(low_range)
        mask = cv2.bitwise_and(mask, mask, mask=low_range_inv)
        mask_fill = (low_range/255 * low_val)
        mask = (mask + mask_fill).astype(np.uint8)

        #misc.imsave('new_mask_out.png', mask_fill)
        #misc.imsave('new_mask.png', mask)

        return mask
