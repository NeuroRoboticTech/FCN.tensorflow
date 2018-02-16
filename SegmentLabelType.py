#!/usr/bin/env python
#
# LabelInfo class
# v1.0.0
#
# LabelInfo class.
#
'''
## License

  COPYRIGHT (C) 2016 NeuroRobotic Technologies, LLC
  ALL RIGHTS RESERVED. This code is not for public use.

'''
import numpy as np
import cv2
import image_utils as iu

class SegmentLabelType(object):
  """ represent a segment label type.
      attributes: label, color, low_color, high_color.
  """

  def calc_color_index(self, new_color, find_color, color_offset, index):
    if find_color[index] + color_offset < 0:
      new_color[index] = 0
    elif find_color[index] + color_offset > 255:
      new_color[index] = 255
    else:
      new_color[index] = find_color[index] + color_offset

  def calc_color(self, find_color, color_offset):
    new_color = np.array([0, 0, 0])
    for offset in range(3):
      self.calc_color_index(new_color, find_color, color_offset, offset)
    
    return new_color

  def __init__(self, label, draw_color, find_color, color_offset, color_number, erode_dialate_mask = True):
    self.label = label
    self.color = draw_color
    self.low_color = self.calc_color(find_color, -color_offset)
    self.high_color = self.calc_color(find_color, color_offset)
    self.color_mask = None
    self.color_number = color_number
    self.erode_dialate_mask = erode_dialate_mask

  def eliminateSmallBlobs(self, img, threshold, mask_color):

    # print img.shape
    # _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print len(contours)

    new_img = np.zeros(img.shape, np.uint8)

    contour_idx = -1
    for cnt in contours:
      area = cv2.contourArea(cnt)
      # print "contour " + str(contour_idx) + " area: " + str(area)
      if area > threshold:  
         # print "drawing contour " + str(contour_idx)
         cv2.drawContours(new_img, contours, contour_idx, mask_color, -1)

      contour_idx = contour_idx + 1    

    return new_img

  def generateColorMask(self, img, threshold):
    # print(self.low_color)
    # print(self.high_color)
    mask = cv2.inRange(img, self.low_color, self.high_color)

    dilate_mask = cv2.dilate(mask,np.ones((3,3),np.uint8))
    erode_mask = cv2.erode(dilate_mask, np.ones((3, 3)))
    final_mask = cv2.dilate(erode_mask,np.ones((2,2),np.uint8))

    pruned_mask = self.eliminateSmallBlobs(final_mask, threshold, 1)

    self.color_mask = np.zeros(img.shape,np.uint8)
    self.color_mask[:,:,0] = pruned_mask * self.color[0]
    self.color_mask[:,:,1] = pruned_mask * self.color[1]
    self.color_mask[:,:,2] = pruned_mask * self.color[2]

  def generateColorMaskFromBWMask(self, bw_mask):

    pixel_mask = np.array([self.color_number,
                           self.color_number,
                           self.color_number])
    mask = cv2.inRange(bw_mask, pixel_mask, pixel_mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    self.color_mask = np.zeros(bw_mask.shape,np.uint8)
    self.color_mask[:,:,0] = mask * self.color[0]
    self.color_mask[:,:,1] = mask * self.color[1]
    self.color_mask[:,:,2] = mask * self.color[2]

    # cv2.imwrite("F:/Projects/FCN_tensorflow/data/Data_zoo/Weeds/test_mask.png", self.color_mask)

  def generateBWMask(self, img, threshold):
    # print(self.low_color)
    # print(self.high_color)
    mask = cv2.inRange(img, self.low_color, self.high_color)

    if self.erode_dialate_mask:
      dilate_mask = cv2.dilate(mask,np.ones((5,5),np.uint8))
      erode_mask = cv2.erode(dilate_mask, np.ones((5, 5)))
      final_mask = erode_mask #cv2.dilate(erode_mask,np.ones((3,3),np.uint8))
    else:
      final_mask = mask

    # print ("Final mask: " + str(final_mask.shape))
    
    if threshold > 0:
      self.color_mask = self.eliminateSmallBlobs(final_mask, threshold, self.color_number)
    else:
      self.bw_mask = np.zeros(img.shape[:2], np.uint8)
      self.bw_mask = ((final_mask/255) * self.color_number).astype(np.uint8)

    #iu.scaleAndShowImage(final_mask, 'final_mask', 0, 0.25)
    #iu.scaleAndShowImage(self.color_mask, 'color_mask', 0, 0.25)

  def __str__(self):
    out =  self.label + \
      ", color: " + str(self.color) + \
      ", low_color: " + str(self.low_color) + \
      ", high_color: " + str(self.high_color)
      
    return out
    

