# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to create a list of all objects in the CRIM data
"""
import itertools
import yaml
import os

with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SHAPES = ['cube', 'sphere', 'cylinder']
MATERIAL = ['metal', 'rubber']

def create_vocab():
    obj_list = list(itertools.product(SHAPES, COLORS, MATERIAL))
    obj_list = [' '.join(x) for x in obj_list]

    os.makedirs(keys['crim_hdf5_folder'], exist_ok=True)

    with open(keys['crim_objects'], 'w') as f:
        for item in obj_list:
            f.write("%s\n" % item)


if __name__ == '__main__':
    create_vocab()
