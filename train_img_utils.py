#!/usr/bin/env python
#
# utilities for checking and producing training images from labeled images.
# v1.0.0
#
#
'''
## License

  COPYRIGHT (C) 2016 NeuroRobotic Technologies, LLC
  ALL RIGHTS RESERVED. This code is not for public use.

'''

# David Cofer
# Initial Date: 11 July 2016
# Last Updated: 11 July 2016
# http://www.NeuroRoboticTech.com/

import os
import argparse
import cv2
import numpy as np
import image_utils
import copy
import math
import image_utils as iu

import SegmentLabelType as SType


# Creates the label list and maps my color codes to pascal voc
def createLabelList():
  offset = 10
  good_insulator_label = SType.SegmentLabelType('good_insulator', np.array([0, 0, 128]), np.array([0,0,255]), offset, 192)

  flashed_insulator_label = SType.SegmentLabelType('flashed_insulator', np.array([0, 128, 128]), np.array([255,0,255]), offset, 192)
  flashed_label = SType.SegmentLabelType('flashed', np.array([128, 0, 0]), np.array([255,150,255]), offset, 128)
  flashed_top_label = SType.SegmentLabelType('flashed_top', np.array([64, 0, 0]), np.array([255,225,255]), offset, 192)

  #broken_insulator_label = SType.SegmentLabelType('broken_insulator', np.array([255,0,0]), np.array([255,0,0]), offset, 0)
  #broken_label = SType.SegmentLabelType('broken', np.array([255,100,100]), np.array([255,100,100]), offset, 0)

  contaminated_insulator_label = SType.SegmentLabelType('contaminated_insulator', np.array([255,220,0]), np.array([255,220,0]), offset, 192)
  #contamination_label = SType.SegmentLabelType('contamination', np.array([255,255,150]), np.array([255,255,150]), offset, 64)

  label_list = [good_insulator_label, flashed_insulator_label, flashed_label, flashed_top_label, contaminated_insulator_label]

  return label_list


def createIgnoreLabelList():
  offset = 10

  broken_insulator_label = SType.SegmentLabelType('broken_insulator', np.array([255, 0, 0]), np.array([255, 0, 0]),
                                                  offset, 255)
  broken_label = SType.SegmentLabelType('broken', np.array([255, 100, 100]), np.array([255, 100, 100]), offset, 255)

  contamination_label = SType.SegmentLabelType('contamination', np.array([255, 255, 150]), np.array([255, 255, 150]),
                                               offset, 255)

  ignore_label = SType.SegmentLabelType('ignore', np.array([192, 224, 224]), np.array([255,255,255]), offset, 255)

  label_list = [ignore_label, broken_insulator_label, broken_label, contamination_label]

  return label_list


def createInsulatorAndFlashedLists():
  label_list = createLabelList()
  ignore_list = createIgnoreLabelList()

  return label_list, ignore_list


# Creates the label list and maps my color codes to pascal voc
def createFlashedOnlyLabelList():
  offset = 10
  good_insulator_label = SType.SegmentLabelType('good_insulator', np.array([0, 0, 128]), np.array([0,0,255]), offset, 128)

  flashed_insulator_label = SType.SegmentLabelType('flashed_insulator', np.array([0, 128, 128]), np.array([255,0,255]), offset, 128)
  flashed_label = SType.SegmentLabelType('flashed', np.array([128, 0, 0]), np.array([255,150,255]), offset, 225)
  flashed_top_label = SType.SegmentLabelType('flashed_top', np.array([64, 0, 0]), np.array([255,225,255]), offset, 128)

  broken_insulator_label = SType.SegmentLabelType('broken_insulator', np.array([255,0,0]), np.array([255,0,0]), offset, 128)
  broken_label = SType.SegmentLabelType('broken', np.array([255,100,100]), np.array([255,100,100]), offset, 128)

  contaminated_insulator_label = SType.SegmentLabelType('contaminated_insulator', np.array([255,220,0]), np.array([255,220,0]), offset, 128)
  contamination_label = SType.SegmentLabelType('contamination', np.array([255,255,150]), np.array([255,255,150]), offset, 128)

  label_list = [good_insulator_label, flashed_insulator_label, flashed_label, flashed_top_label,
                broken_insulator_label, broken_label, contaminated_insulator_label, contamination_label]

  return label_list


def createFlashedOnlyIgnoreLabelList():
  offset = 10

  #broken_insulator_label = SType.SegmentLabelType('broken_insulator', np.array([255, 0, 0]), np.array([255, 0, 0]),
  #                                                offset, 255)
  #broken_label = SType.SegmentLabelType('broken', np.array([255, 100, 100]), np.array([255, 100, 100]), offset, 255)

  #contamination_label = SType.SegmentLabelType('contamination', np.array([255, 255, 150]), np.array([255, 255, 150]),
  #                                             offset, 255)

  ignore_label = SType.SegmentLabelType('ignore', np.array([192, 224, 224]), np.array([255,255,255]), offset, 255)

  label_list = [ignore_label]

  return label_list


def createFlashedOnlyLists():
  label_list = createFlashedOnlyLabelList()
  ignore_list = createFlashedOnlyIgnoreLabelList()

  return label_list, ignore_list


def generateBWLabelMask(img, labels, ignore_labels, threshold):
  # create a combined mask with all zeros the same size as image
  # so we can combine all the masks together
  combined_mask = np.zeros(img.shape[:2], np.uint8)
  combined_ignore_mask = np.zeros(img.shape[:2], np.uint8)

  # Go through and generate the mask for each color and add it to the
  for label_type in labels:
    label_type.generateBWMask(img, threshold)
    combined_mask = combined_mask + label_type.bw_mask

  for label_type in ignore_labels:
    label_type.generateBWMask(img, threshold)
    combined_ignore_mask = combined_ignore_mask + label_type.bw_mask

  ignore_mask = cv2.bitwise_not(combined_ignore_mask)
  ignore_mask_bgr = cv2.merge((ignore_mask, ignore_mask, ignore_mask)).astype(np.uint8)

  return combined_mask, ignore_mask_bgr


def generateColorLabelMaskFromBWMask(bw_mask, labels):
  # create a combined mask with all zeros the same size as image
  # so we can combine all the masks together
  combined_mask = np.zeros(bw_mask.shape, np.uint8)

  # Go through and generate the mask for each color and add it to the
  for label_type in labels:
    label_type.generateColorMaskFromBWMask(bw_mask)
    combined_mask = combined_mask + label_type.color_mask

  return combined_mask


