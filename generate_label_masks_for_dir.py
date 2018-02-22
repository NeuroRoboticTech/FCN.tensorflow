from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import cv2
import image_utils as iu
import train_img_utils as tu
import os

#color_img_dirs = ["C:/Work/PowerLineInspection/aeryon_database/Defective/Flashed Insulators/",
#                  "C:/Work/PowerLineInspection/aeryon_database/NonDefective/"
#                 ]

color_img_dirs = ["C:/Work/PowerLineInspection/aeryon_database/Defective/Flashed Insulators/"
                 ]

#color_img_dirs = ["C:/Work/PowerLineInspection/aeryon_database/test/"]
image_dir = "C:/Work/PowerLineInspection/aeryon_database/images/"
mask_dir = "C:/Work/PowerLineInspection/aeryon_database/labels/"

label_list, ignore_list = tu.createFlashedOnlyLists()


def process_dir(input_dir):
  colored_images = iu.findFilesOfType(input_dir, ["_mask.JPG", "_mask.jpg"])

  for img_name in colored_images:
    base_img_name = img_name[:-9] + ".JPG"
    print("Processing: " + input_dir + base_img_name)

    already_proc_files = iu.findFilesContaining(image_dir, [img_name[:-9]])

    if len(already_proc_files) <= 0:
        if os.path.isfile(input_dir  + base_img_name):
          print("Generating label mask")
          # Load in image and pad it out so it is N x min_width, N x min_height
          color_img = cv2.imread(input_dir + img_name)

          # print(color_img.shape)
          # mask = tu.generateColorLabelMask(color_img, label_list, 200)
          mask, ignore_mask = tu.generateBWLabelMask(color_img, label_list, ignore_list, 0)

          # Load up the original image
          orig_img = cv2.imread(input_dir + base_img_name)
          orig_img_without_ignore = ignore_mask & orig_img #cv2.bitwise_and(orig_img, ignore_mask)

          # Save out original images
          img_out_name = image_dir + base_img_name
          cv2.imwrite(img_out_name, orig_img_without_ignore)
          #img_ignore_name = image_dir + "ignore_" + base_img_name
          #cv2.imwrite(img_ignore_name, ignore_mask)
          label_out_name = mask_dir + base_img_name[:-4] + ".png"
          cv2.imwrite(label_out_name, mask)
          print("saved image: " + img_out_name)

        else:
          print("Missing: " + base_img_name)
    else:
        print("Skipping file already processed: " + base_img_name)


for input_dir in color_img_dirs:
  process_dir(input_dir)