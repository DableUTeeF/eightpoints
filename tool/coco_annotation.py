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
        candit = {'top_left': [(1e7, 1e7), float('inf')],
                  'top': [[-1, float('inf')]],
                  'top_right': [(1e7, -1e7), float('inf')],
                  'right': [[-float('inf'), 1]],
                  'bot_right': [(-1e7, -1e7), float('inf')],
                  'bot': [[-1, -float('inf')]],
                  'bot_left': [(-1e7, 1e7), float('inf')],
                  'left': [[float('inf'), 1]],
                  }
        vertical, horizontal, obtuse, quirk, center = get_lines(x, y, x2, y2, w, h)

        v = np.array(vertical)
        # v[:, 0] -= x
        # v[:, 1] -= y
        v = v.astype('uint16')
        v_top = v[0].tolist()
        v_bot = v[1].tolist()

        h = np.array(horizontal)
        # h[:, 0] -= x
        # h[:, 1] -= y
        h = h.astype('uint16')
        h_left = h[0].tolist()
        h_right = h[1].tolist()

        o = np.array(obtuse)
        # o[:, 0] -= x
        # o[:, 1] -= y
        o = o.astype('uint16').tolist()

        q = np.array(quirk)
        # q[:, 0] -= x
        # q[:, 1] -= y
        q = q.astype('uint16').tolist()

        mask = coco.annToMask(ant) * 255
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

            # top - bot
            if point[0] == v_top[0]:  # todo: 26/11. better try to visuzlize here
                if point[1] < candit['top'][0][1]:
                    d = abs(int(point[1] - center[1]))
                    candit['top'] = (point, d)
            if point[0] == v_bot[0]:
                if point[1] > candit['bot'][0][1]:
                    d = abs(int(point[1] - center[1]))
                    candit['bot'] = (point, d)

            # left - right
            if point[1] == h_left[1]:
                if point[0] < candit['left'][0][0]:
                    d = abs(int(point[0] - center[0]))
                    candit['left'] = (point, d)
            if point[1] == h_right[1]:
                if point[0] > candit['right'][0][0]:
                    d = abs(int(point[0] - center[0]))
                    candit['right'] = (point, d)

            # top_left - bot_right
            distance = distance_from_line(o, point)
            if distance < candit['top_left'][1] and distance < 1e-2:
                if sum(point) < sum(candit['top_left'][0]):
                    d = np.sqrt(((point[0] - center[0])**2 + ((point[1] - center[1])**2)))
                    d = int(d)
                    candit['top_left'] = (point, distance, d)
            if distance < candit['bot_right'][1] and distance < 1e-2:
                if sum(point) > sum(candit['bot_right'][0]):
                    d = np.sqrt(((point[0] - center[0])**2 + ((point[1] - center[1])**2)))
                    d = int(d)
                    candit['bot_right'] = (point, distance, d)

            # top_right - bot_left
            distance = distance_from_line(q, point)
            if distance < candit['top_right'][1] and distance < 1e-2:
                if point[0] < candit['top_right'][0][0]:
                    d = np.sqrt(((point[0] - center[0])**2 + ((point[1] - center[1])**2)))
                    d = int(d)
                    candit['top_right'] = (point, distance, d)
            if distance < candit['bot_left'][1] and distance < 1e-2:
                if point[0] > candit['bot_left'][0][0]:
                    d = np.sqrt(((point[0] - center[0])**2 + ((point[1] - center[1])**2)))
                    d = int(d)
                    candit['bot_left'] = (point, distance, d)

        name_box_id[name].append([ant['bbox'], cat, candit])

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

                try:
                    box_info = " %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, int(info[1]),
                                                                            info[2]['top_left'][2], info[2]['top'][1], info[2]['top_right'][2],
                                                                            info[2]['right'][1], info[2]['bot_right'][2], info[2]['bot'][1],
                                                                            info[2]['bot_left'][2], info[2]['left'][1]
                                                                            )
                    f.write(box_info)
                except IndexError:
                    print(key)
            f.write('\n')
