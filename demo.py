import argparse
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
from dataset.voc import BaseTransform
from dataset.config import MEANS
from dataset.config import VOC_CLASSES as labelmap
from utils.dirs import create_dir
from utils.draw_boxes import draw_boxes
from ssd.ssd300 import build_ssd

parser = argparse.ArgumentParser(description='Single image prediction.')
parser.add_argument('--visualize', default=True, type=bool, help='is visualize on test')
parser.add_argument('--output_dir', default='./test_output', type=str, help='save path')
args = parser.parse_args()


def main(images_list, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = BaseTransform(300, MEANS)

    net = build_ssd('test', 300, 21).to(device)
    net.load_state_dict(torch.load('./weights/ssd300_mAP_77.43_v2.pth', map_location="cpu"))
    net.eval()

    with torch.no_grad():
        for img in images_list:
            image = cv2.imread(img)
            h, w, _ = image.shape
            scale = torch.from_numpy(np.array([w, h, w, h])).float()

            x, _, _ = transform(image)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
            y = net(x)

            for i in range(1, y.size(1)):
                idx = y[0, i, :, 0] >= 0.6
                detections = y[0, i, idx].view(-1, 5)
                if detections.numel() == 0:
                    continue
                scores, locs = detections[:, 0], (detections[:, 1:] * scale).cpu().numpy()
                label_name = labelmap[i - 1]
                for j in range(len(scores)):
                    display_txt = '{}: {:.2f}'.format(label_name, scores[j])
                    draw_boxes(image, locs[j], display_txt, j)

            if args.visualize:
                cv2.imshow('detect_result', image)
                cv2.waitKey()
            cv2.imwrite(os.path.join(save_path, 'result_' + img.split('/')[-1]), image)


if __name__ == '__main__':
    image_list = ['./dataset/xiaomi.jpg']
    create_dir(args.output_dir)
    main(image_list, args.output_dir)
