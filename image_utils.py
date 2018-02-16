#!/usr/bin/env python
#
# image utility functions
# v1.0.0
#
# Some simple image utility functions
#


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def findFilesOfType(input_dir, endings): 
    # Get the xml files in the directory
    included_extenstions = endings
    img_files_list = [fn for fn in os.listdir(input_dir)
                      if any(fn.endswith(ext) for ext in included_extenstions)]
    img_files = sorted(set(img_files_list))
    # print img_files

    return img_files


def findFilesContaining(input_dir, searches):
    # Get the xml files in the directory
    included_extenstions = searches
    img_files_list = []
    for fn in os.listdir(input_dir):
        for ext in included_extenstions:
            if ext in fn and os.path.isfile(os.path.join(input_dir, fn)):
                img_files_list.append(fn)
    # img_files_list = [fn for fn in os.listdir(input_dir)
    #                  if ext in fn for ext in included_extenstions]
    img_files = sorted(set(img_files_list))
    # print img_files

    return img_files


def findFilesOfTypeWithExclude(input_dir, endings, exclude_list):
    # Get the xml files in the directory
    included_extenstions = endings
    img_files_list = [fn for fn in os.listdir(input_dir)
                      if any(fn.endswith(ext) for ext in included_extenstions) and fn not in exclude_list]
    img_files = sorted(set(img_files_list))
    # print img_files

    return img_files


def scaleAndShowImage(img, name='image', wait=0, scale=1):
    res_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(name, res_img)
    cv2.waitKey(wait)    


def warpImageToMatch(input_img, scale, x_offset, y_offset, rotate, out_width, out_height):
  # First scale the image
  scale_img = cv2.resize(input_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

  (scale_h, scale_w) = scale_img.shape[:2]

  # now rotate image if needed
  if rotate != 0:
    # print rotate
    scale_center = (scale_w / 2, scale_h / 2)
    rotateMat = cv2.getRotationMatrix2D(scale_center, rotate, 1.0)
    rotated_img = cv2.warpAffine(scale_img, rotateMat, (scale_w, scale_h))
    # cv2.imshow("rotated Image", rotated_image)
  else:
    rotated_img = scale_img

  transM = np.float32([[1,0,x_offset],[0,1,y_offset]])
  out_img = cv2.warpAffine(rotated_img,transM,(out_width, out_height))

  return out_img


def calculateRGBCorrectionMatrix(r, g, b, powers):

  preproc = np.zeros([r.shape[0], r.shape[1], powers.shape[0]])

  idx = 0
  for power in powers:
    # print("power: ", power)
    preproc[:,:, idx] = (np.power(r, power[0]) * np.power(g, power[1]) * np.power(b, power[2]))
    # print(preproc[:,:, idx])
    idx = idx + 1

  # print(preproc)
  return preproc


# Resizes based only on width. See resize_img_label_fixed_size for more details.
def resize_img_label_fixed_min_width(img, label, min_width, min_height):
  img_width = img.shape[1]
  img_height = img.shape[0]

  # If the image is already larger than the min width then just
  # use the existing image.
  if img_width >= min_width and img_height >= min_height:
    return img, label
  else:
    width_ratio = float(min_width) / float(img_width)
    new_height = int(img_height * width_ratio)

    # If the height is still too small then resize again to make min height.
    if new_height < min_height:
      height_ratio = float(min_height) / float(new_height)
      new_width = int((min_width * height_ratio) + 0.5)
      new_height = min_height
      min_width = new_width

    return resize_img_label_fixed_size(img, label, min_width, new_height)


# Takes an image and a mask label image and resizes them to a specified size.
# It uses nearest neighbor interpolation on the mask to ensure that no interpolation
# is used on the new pixel values.
def resize_img_label_fixed_size(img, label, width, height):

  # First resize the image itself
  img_width = img.shape[1]
  img_height = img.shape[0]
  if img_width > width or img_height > height:
    # Shrinking image so use inter area
    res_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
  else:
    # Enlarging image so use inter cubic
    res_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

  res_label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)

  return res_img, res_label

