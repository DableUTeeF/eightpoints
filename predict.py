from models import Yolov4
import torch
from tool.utils import post_processing, plot_boxes_cv2
from dataset import resize_image
import cv2
import numpy as np

def plot_lines(image, boxes):
    width = img.shape[1]
    height = img.shape[0]
    angled = np.sqrt(width ** 2 + height ** 2)

    for box in boxes:
        x = (box[2] + box[0]) / 2 * width
        y = (box[3] + box[1]) / 2 * height
        print()

    return image

if __name__ == '__main__':
    model = Yolov4(None, n_classes=80, inference=True)
    state = torch.load('16.pth')
    model.load_state_dict(state['model'])
    del state
    model.eval()

    rawimg = cv2.imread('/media/palm/data/coco/images/val2017/000000289343.jpg')
    rawimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
    rawimg, bboxes = resize_image(rawimg, None, 608, 608)
    img = rawimg.copy().transpose(2, 0, 1)

    inputs = torch.from_numpy(np.expand_dims(img, 0).astype('float32')).div(255.0)
    output = model(inputs)
    boxes = post_processing(img, 0.4, 0.4, output)
    img = plot_boxes_cv2(rawimg.astype('uint8'), boxes[0])
    img = plot_lines(img, boxes[0])
    cv2.imshow('a', img)
    cv2.waitKey()


