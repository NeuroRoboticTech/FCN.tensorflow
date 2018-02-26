import numpy as np
import image_utils as iu
import os

input_dir = "C:\\Work\\FCN.tensorflow\\Data_Zoo\\Insulators\\data\\annotations\\validation"
colored_images = iu.findFilesOfType(input_dir, ["_mask.png"])

for img_name in colored_images:
    new_img_name = img_name[:-9] + ".png"
    os.rename(input_dir + "\\" + img_name, input_dir + "\\" + new_img_name)
