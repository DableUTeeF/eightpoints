import numpy as np
import cv2
import os


def main():
    img = np.zeros((399, 399, 3))
    point = (100, 300)
    center = (200, 200)
    d = np.sqrt(((point[0] - center[0]) ** 2 + ((point[1] - center[1]) ** 2)))
    d = int(d)
    angle = np.arctan2(center[1] - point[1], point[0] - center[0]) * 180 / np.pi
    angle = (angle + 360) % 360
    print('d', d)
    print('y', center[1] - point[1])
    print('x', point[0] - center[0])
    print(angle)
    img = cv2.line(img, point, center, (255, 255, 0))
    cv2.imshow('a', img)
    cv2.waitKey()

def resize_image(img, w, h):  # well, we're gonna make it square anyway, but I'm using both w and h for th time being
    out = np.zeros((h, w, 3))
    ih, iw = img.shape[:-1]
    if iw > ih:
        img = cv2.resize(img, (w, int(ih / (iw / w))))
        scale = w / iw
    else:
        img = cv2.resize(img, (int(iw / (ih/ h)), h))
        scale = h / ih
    out[:img.shape[0], :img.shape[1]] = img
    return out, scale

if __name__ == '__main__':
    csv = open('/home/palm/PycharmProjects/pytorch-YOLOv4/data/val.txt').readlines()
    dataset_dir = '/media/palm/data/coco/images'
    for line in csv:
        line = line[:-1].split(' ')
        image = cv2.imread(os.path.join(dataset_dir, line[0]))
        image, scale = resize_image(image, 608, 608)
        for box in line[1:]:
            box = box.split(',')
            # image = cv2.rectangle(image, (int(box[0]), int(box[1])),
            #                       (int(box[2]), int(box[3])),
            #                       (255, 255, 255))

            x = int((int(box[2]) + int(box[0])) / 2 * scale)
            y = int((int(box[3]) + int(box[1])) / 2 * scale)
            top = int(box[5])
            top_left = int(box[6])
            top = int(top * scale)
            top_left = int(top_left * scale)
            image = cv2.line(image, (x, y), (int(x - top * 0.7), int(y - top * 0.7)),
                             (255, 255, 0), 2)

            image = cv2.line(image, (x, y), (x, y - top_left),
                             (0, 255, 0), 2)

        cv2.imshow('a', image.astype('uint8'))
        cv2.waitKey()
