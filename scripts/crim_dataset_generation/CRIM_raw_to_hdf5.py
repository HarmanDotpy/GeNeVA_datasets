# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to parse and read raw i-CLEVR data and save it in HDF5 format for GeNeVA-GAN
"""
from glob import glob
import json
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def create_h5():
    # load required keys
    data_path = keys['crim_data_source']
    output_path = keys['crim_hdf5_folder']
    OBJECTS = keys['crim_objects']
    with open(OBJECTS, 'r') as f:
        OBJECTS = f.readlines()
        OBJECTS = [tuple(x.strip().split()) for x in OBJECTS]

    # create hdf5 files for train, val, test
    train_h5 = h5py.File(os.path.join(output_path, 'clevr_train.h5'), 'w')
    val_h5 = h5py.File(os.path.join(output_path, 'clevr_val.h5'), 'w')
    test_h5 = h5py.File(os.path.join(output_path, 'clevr_test.h5'), 'w')

    # add object properties to hdf5
    entites = json.dumps(['{} {} {}'.format(e[0], e[1], e[2]) for e in OBJECTS])
    train_h5.create_dataset('entities', data=entites)
    val_h5.create_dataset('entities', data=entites)
    test_h5.create_dataset('entities', data=entites)

    h5_objects = {'train': train_h5, 'valid':val_h5, 'test':test_h5}

    for split in ['train', 'valid', 'test']:
                
        # make imagenames2command dictionary, images come from images_c1 folder
        im2comm = dict()
        with open(os.path.join(data_path, 'CLEVR_questions.json'), 'r') as f:
            command_dicts = json.load(f)['questions']
            im2comm = {command_dict['image_output_filename']: command_dict['question'] for  command_dict in command_dicts}

        # use the data_path to make a list of images/, images_c1/
        images = [os.path.basename(x) for x in tqdm(sorted(glob(os.path.join(data_path, split, 'images', 'CLEVR_new*'))))] # the images before change command
        images_c1 = [os.path.basename(x) for x in tqdm(sorted(glob(os.path.join(data_path, split, 'images_c1', 'CLEVR_new*'))))] # the images after change command
        scenes = [os.path.basename(x) for x in tqdm(sorted(glob(os.path.join(data_path, split, 'scenes', 'CLEVR_new*'))))] # the scenes before change command
        scenes_c1 = [os.path.basename(x) for x in tqdm(sorted(glob(os.path.join(data_path, split, 'scenes_c1', 'CLEVR_new*'))))] # the scenes after change command
        assert len(images)==len(scenes)
        assert len(images_c1)==len(scenes_c1)

        # combine the above images and make tuples
        command_data = [] # of the form [{'im1': img, 'im2': changed_image_name, 'scene1': scenes[img_idx], 'scene2': scenes_c1[images_c1_counter], 'command': im2comm[changed_image_name]}, ....]
        images_c1_counter = 0
        for img_idx, img in enumerate(images):
            image_name = img.split('.')[0] # eg CLEVR_new_000041
            while images_c1_counter<len(images_c1) and image_name in images_c1[images_c1_counter]:
                changed_image = images_c1[images_c1_counter]
                command_data.append({'im1': img, 'im2': changed_image, 'scene1': scenes[img_idx], 'scene2': scenes_c1[images_c1_counter], 'command': im2comm[changed_image]})
                images_c1_counter+=1
                # import pdb; pdb.set_trace()

        assert len(command_data) == len(images_c1)

        for data_dict in tqdm(command_data):
            img1_path = os.path.join(data_path, split, 'images/', data_dict['im1'])
            img2_path = os.path.join(data_path, split, 'images_c1/', data_dict['im2'])
            scene1_path = os.path.join(data_path, split, 'scenes/', data_dict['scene1'])
            scene2_path = os.path.join(data_path, split, 'scenes_c1/', data_dict['scene2'])

            with open(scene1_path, 'r') as f:
                scene1 = json.load(f)

            with open(scene2_path, 'r') as f:
                scene2 = json.load(f)
            scene_id = os.path.basename(scene2_path).split('_', 2)[2][:-5] # these are unique, for eg CLEVR_new_001991_0_c1.json

            # add text
            text = [data_dict['command']] # 1 element list since contrary to geneva, we have just one command and geneva has 5
            
            # add images
            image1 = cv2.resize(cv2.imread(img1_path), (128, 128))
            image2 = cv2.resize(cv2.imread(img2_path), (128, 128))
            images = [image1, image2]


            # add objects and object coordinates

            objects = np.zeros((2, 48)) # 2 since we have only 2 images in CRIM
            object_coords = np.zeros((2, 48, 3))

            # Objects for scene 1 
            agg_object = np.zeros(48) # 48 because there are 48 kinds of objects that we have
            agg_object_coords = np.zeros((48, 3))
            for t, obj in enumerate(scene1['objects']):
                color = obj['color']
                shape = obj['shape']
                material = obj['material']
                index = OBJECTS.index((shape, color, material))
                agg_object[index] = 1

                agg_object_coords[index] = [obj['pixel_coords'][0]/480.*128, obj['pixel_coords'][1]/320.*128, obj['pixel_coords'][2]]
                
            object_coords[1] = agg_object_coords
            objects[0] = agg_object
            
            # Objects for scene 2
            agg_object = np.zeros(48) # 48 because there are 48 kinds of objects that we have
            agg_object_coords = np.zeros((48, 3))
            for t, obj in enumerate(scene2['objects']):
                color = obj['color']
                shape = obj['shape']
                material = obj['material']
                index = OBJECTS.index((shape, color, material))
                agg_object[index] = 1

                agg_object_coords[index] = [obj['pixel_coords'][0]/480.*128, obj['pixel_coords'][1]/320.*128, obj['pixel_coords'][2]]
                
            object_coords[1] = agg_object_coords
            objects[1] = agg_object

            sample = h5_objects[split].create_group(scene_id)

            sample.create_dataset('scene_id', data=scene_id)
            sample.create_dataset('images', data=np.array(images))
            sample.create_dataset('text', data=json.dumps(text))
            sample.create_dataset('objects', data=objects)
            sample.create_dataset('coords', data=np.array(object_coords))           

            # import pdb; pdb.set_trace()

if __name__ == '__main__':
    create_h5()
