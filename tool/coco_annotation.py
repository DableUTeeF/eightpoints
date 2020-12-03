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
import numpy as np
from pycocotools.coco import COCO
import cv2


def distance_from_line(line, point):
    distance_1 = np.sqrt((line[0][0] - point[0]) ** 2 + (line[0][1] - point[1]) ** 2)
    distance_2 = np.sqrt((line[1][0] - point[0]) ** 2 + (line[1][1] - point[1]) ** 2)
    line_length = np.sqrt((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2)
    return distance_1 + distance_2 - line_length


def get_lines(x1, y1, x2, y2, w, h):
    center = (x1 + w / 2, y1 + h / 2)
    horizontal = ((x1, y1 + h / 2), (x2, y1 + h / 2))
    vertical = ((x1 + w / 2, y1), (x1 + w / 2, y2))
    obtuse = ((x1, y1), (x2, y2))
    quirk = ((x1, y2), (x2, y1))
    return vertical, horizontal, obtuse, quirk, center


if __name__ == '__main__':
    """hyper parameters"""
    json_file_path = '/media/palm/data/coco/annotations/instances_train2017.json'
    images_dir_path = 'train2017'
    output_path = '../data/train.txt'
    coco = COCO(json_file_path)

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
        x, y, w, h = ant['bbox']
        x2 = x + w
        y2 = y + h
        candit = {'top_left': [float('inf'), 1],  # todo: change this to list will allow using more points
                  'top': [float('inf'), 1],
                  'top_right': [float('inf'), 1],
                  'right': [float('inf'), 1],
                  'bot_right': [float('inf'), 1],
                  'bot': [float('inf'), 1],
                  'bot_left': [float('inf'), 1],
                  'left': [float('inf'), 1],
                  }
        vertical, horizontal, obtuse, quirk, center = get_lines(x, y, x2, y2, w, h)

        mask = coco.annToMask(ant) * 255
        img = mask.copy().astype('uint8')
        ret, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contour) == 0:
            continue
        cnt = contour[0]
        cnt = np.squeeze(cnt)
        if len(cnt.shape) == 1:
            cnt = np.expand_dims(cnt, 0)

        for point in cnt:
            # point[0] -= x
            # point[1] -= y
            point = point.tolist()
            angle = np.arctan2(center[1] - point[1], point[0] - center[0]) * 180 / np.pi
            angle = (angle + 360) % 360

            # top_left: 135
            if abs(angle - 135) < candit['top_left'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['top_left'] = (abs(angle - 135), point, distance)

            # top: 90
            if abs(angle - 90) < candit['top'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['top'] = (abs(angle - 90), point, distance)

            # top_right: 45
            if abs(angle - 45) < candit['top_right'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['top_right'] = (abs(angle - 45), point, distance)

            # right: 0
            if abs(angle - 0) < candit['right'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['right'] = (abs(angle - 0), point, distance)

            # bot_right: 315
            if abs(angle - 315) < candit['bot_right'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['bot_right'] = (abs(angle - 315), point, distance)

            # bot: 270
            if abs(angle - 270) < candit['bot'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['bot'] = (abs(angle - 270), point, distance)

            # bot_left: 225
            if abs(angle - 225) < candit['bot_left'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['bot_left'] = (abs(angle - 225), point, distance)

            # left: 180
            if abs(angle - 180) < candit['left'][0]:
                distance = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
                distance = int(distance)
                candit['left'] = (abs(angle - 180), point, distance)

        name_box_id[name].append([ant['bbox'], cat, candit])

    """write to txt"""
    with open(output_path, 'w') as f:
        for key in tqdm(name_box_id.keys()):
            f.write(key)
            box_infos = name_box_id[key]
            # image = cv2.imread(os.path.join('/media/palm/data/coco/images', key))
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])
                # x = int(x_min + x_max) // 2
                # y = int(y_min + y_max) // 2
                # image = cv2.line(image, (x, y), (int(info[2]['top_left'][1][0]), int(info[2]['top_left'][1][1])),
                #                  (0, 255, 0))
                #
                # d = info[2]['top_left'][2]
                # image = cv2.line(image, (x, y), (int(x - int(d) * 0.7), int(y - int(d) * 0.7)),
                #                  (0, 0, 255))
                # cv2.imshow('a', image)
                try:
                    box_info = " %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, int(info[1]),
                                                                            info[2]['top_left'][2], info[2]['top'][2], info[2]['top_right'][2],
                                                                            info[2]['right'][2], info[2]['bot_right'][2], info[2]['bot'][2],
                                                                            info[2]['bot_left'][2], info[2]['left'][2]
                                                                            )
                    f.write(box_info)
                except IndexError as e:
                    print(key, e)
                cv2.waitKey()
            f.write('\n')
