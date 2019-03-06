import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data

from dataset.config import VOC_CLASSES


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    return torch.stack(imgs, 0), targets


class BaseTransform(object):
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image_ = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image_ -= self.mean
        image_ = image_.astype(np.float32)

        return image_, boxes, labels


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
        Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object"""

    def __init__(self, root, image_sets=(('2007', 'trainval'), ('2012', 'trainval')), transform=None,
                 target_transform=VOCAnnotationTransform(), dataset_name='VOC2007'):
        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.anno_path = os.path.join('%s', 'Annotations', '%s.xml')
        self.img_path = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            root_path = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((root_path, line.strip()))

    def __getitem__(self, index):
        img, gt, h, w = self.pull_item(index)

        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self.anno_path % img_id).getroot()
        img = cv2.imread(self.img_path % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]

        return cv2.imread(self.img_path % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self.anno_path % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)

        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)