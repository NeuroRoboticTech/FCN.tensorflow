#!/usr/bin/env python

import unittest.mock as mock
import unittest
import BatchDatsetReader as dataset
import read_MITSceneParsingData as scene_parsing
import FCN_env_vars as EnvVars

def swapRecords(racords, idx1, idx2):
    temp = racords[idx1]
    racords[idx1] = racords[idx2]
    racords[idx2] = temp

print("test") #EnvVars.data_dir
train_records, valid_records = \
    scene_parsing.read_dataset(EnvVars.data_dir + "\\Data_Zoo\\Insulators", False)
print(len(train_records))

#Set the next images to be loaded.
swapRecords(train_records, 0, 61)

image_options = {'resize': False,
                 'image_height': 380,
                 'image_width': 672,
                 'image_channels': 3}
allowed_mask_vals = [0, 128, 192]
train_dataset_reader = dataset.BatchDatset(
    train_records, 1, allowed_mask_vals, image_options)

#test_image = train_dataset_reader.transform(EnvVars.data_dir + "\\Data_Zoo\\Insulators\\data\\images\\training\\160041_a.JPG")

train_dataset_reader.start()
# Wait for first images to load
train_dataset_reader.wait_for_images()


train_images, train_annotations, train_image_names = \
    train_dataset_reader.next_batch(True, False, force_size_idx=6, force_flip=1, force_rot=-12, force_cut_x=363, force_cut_y=0)

if train_dataset_reader is not None:
    train_dataset_reader.exit_thread = True

