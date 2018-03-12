import numpy as np
import image_utils as iu
import os
import cv2

original_images_dir = "C:\\Work\\PowerLineInspection\\demo_images\\FlashedOnly\\validation"
original_images = iu.findFilesOfType(original_images_dir, ["_input.png"])

flashed_only_dir = "C:\\Work\\PowerLineInspection\\demo_images\\validation_FlashedOnly_RunThree"
flashed_images = iu.findFilesOfType(flashed_only_dir, ["_pred.png"])

insulator_only_dir = "C:\\Work\\PowerLineInspection\\demo_images\\FlashedAndGood\\validation"
insulator_images = iu.findFilesOfType(flashed_only_dir, ["_pred.png"])

out_folder = "C:\\Work\\PowerLineInspection\\demo_images\\combined\\validation2"

if len(original_images) != len(flashed_images) or len(original_images) != len(insulator_images):
    raise RuntimeError("mismatch in number of images")

for idx in range(0, len(original_images)):
    orig_img = cv2.imread(original_images_dir + "\\" + original_images[idx])
    flashed_img = cv2.imread(flashed_only_dir + "\\" + flashed_images[idx], 0)
    ins_img = cv2.imread(insulator_only_dir + "\\" + insulator_images[idx], 0)

    #iu.scaleAndShowImage(orig_img, 'orig image')
    #iu.scaleAndShowImage(flashed_img, 'flashed image')
    #iu.scaleAndShowImage(ins_img, 'insulator image')

    flashed_mask = cv2.inRange(flashed_img, 30, 255)
    #iu.scaleAndShowImage(flashed_mask, 'flashed_mask')

    inv_flashed_mask = cv2.bitwise_not(flashed_mask)
    ins_mask = cv2.inRange(ins_img, 30, 255)
    #iu.scaleAndShowImage(ins_mask, 'insulator mask 1')

    # Remove the flashed portion of the insulator from this mask.
    ins_mask = ins_mask & inv_flashed_mask
    #iu.scaleAndShowImage(ins_mask, 'insulator mask 2')

    zero_matrix = np.zeros(flashed_img.shape, dtype=np.uint8)

    # Flashed as red
    mask_img = cv2.merge((zero_matrix, ins_mask, flashed_mask))
    #iu.scaleAndShowImage(mask_img, 'mask_img')

    combined_img = cv2.addWeighted(orig_img, 1.0, mask_img, 0.35, 0.0)

    #iu.scaleAndShowImage(combined_img, 'combined_img')

    original_file_name = original_images[idx][:-10] + ".png"
    new_file_name = original_images[idx][:-10] + "_combined.png"
    mask_file_name = original_images[idx][:-10] + "_mask.png"
    cv2.imwrite(out_folder + "\\" + original_file_name, orig_img)
    cv2.imwrite(out_folder + "\\" + new_file_name, combined_img)
    cv2.imwrite(out_folder + "\\" + mask_file_name, mask_img)

