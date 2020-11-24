# -*- coding: utf-8 -*-
"""
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

"""
import json
from collections import defaultdict
from tqdm import tqdm
import os

"""hyper parameters"""
json_file_path = '/media/palm/data/coco/annotations/instances_val2017.json'
images_dir_path = 'val2017'
output_path = '../data/val.txt'

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

"""generate labels"""
images = data['images']
annotations = data['annotations']
for ant in tqdm(annotations):
    id = ant['image_id']
    # name = os.path.join(images_dir_path, images[id]['file_name'])
    name = os.path.join(images_dir_path, '{:012d}.jpg'.format(id))
    cat = ant['category_id']

    if 1 <= cat <= 11:
        cat = cat - 1
    elif 13 <= cat <= 25:
        cat = cat - 2
    elif 27 <= cat <= 28:
        cat = cat - 3
    elif 31 <= cat <= 44:
        cat = cat - 5
    elif 46 <= cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif 72 <= cat <= 82:
        cat = cat - 10
    elif 84 <= cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])

"""write to txt"""
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
