#!/usr/bin/env python
#
# Creates a folder structure that mimics the ADEChallengeData2016 data folder
# structure. It has images and annotations folder, and under each it has training
# and validation folders. This also randomly selects X% of the images to be used
# as validation.
# v1.0.0
#
#

import os
import image_utils as iu
from random import randint

in_images_path = "C:/Work/PowerLineInspection/aeryon_database/images/"
in_labels_path = "C:/Work/PowerLineInspection/aeryon_database/labels/"

output_path = "C:/Work/PowerLineInspection/FCN.tensorflow/Data_Zoo/Insulators/flashed_data/"

#Percentage for whether an image should go to validation folder or not.
val_perc = 10

# First get a list of all image files.
imgs = iu.findFilesOfType(in_images_path, [".jpg", ".JPG"])

# Loop through the list of image files
for img_name in imgs:
  base_img_name = img_name[:-4]
  annot_img_path = in_labels_path + base_img_name + ".png"
  img_path = in_images_path + img_name

  # Make sure that there is a matching annotation
  if os.path.isfile(annot_img_path):

    rnd_val = randint(1, 100)
    # if random number is less than validation percentage then
    # place it in the val folder, otherwise put it in the train
    # folder.
    if rnd_val <= val_perc:
      out_img_path = output_path + "images/validation/" + img_name
      out_annot_path = output_path + "annotations/validation/" + base_img_name + ".png"
    else:
      out_img_path = output_path + "images/training/" + img_name
      out_annot_path = output_path + "annotations/training/" + base_img_name + ".png"

    os.rename(img_path, out_img_path)
    os.rename(annot_img_path, out_annot_path)

    print("Moved file: " + base_img_name)



