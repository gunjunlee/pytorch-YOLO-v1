import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision import transforms

from PIL import Image
import numpy as np

import os
import os.path as osp


from dataloader import BaseDataset


class COCODataset(BaseDataset):
    def __init__(self, S, B, C, label_dict, img_root, ann_path):
        self.S = S
        self.B = B
        self.C = C
        self.label_dict = label_dict
        self.img_root = img_root
        self.catIds = [1, 3, 6, 8, 10, 15, 17, 27, 31, 44, 47, 49, 51, 62, 63, 67, 73, 77, 84, 85]
        
        from pycocotools.coco import COCO
        self.coco = COCO(ann_path)
        self.cats = self.coco.loadCats(self.catIds)
        self.imgIds = self.coco.getImgIds()

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        labels, bboxes = self.get_bboxes(idx, img.size)
        target = self.make_target(labels, bboxes)
        # pdb.set_trace()
        # plt.imshow(visualize(img, target, self.label_dict))
        # plt.show()
        
        img = transforms.Resize((448, 448))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        target = torch.tensor(target).float()
        return img, target

    def get_image(self, idx):
        img_path = osp.join(self.img_root, '{:012d}.jpg'.format(self.imgIds[idx]))
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img

    def get_bboxes(self, idx, size):
        bboxes = []
        labels = []
        annIds = self.coco.getAnnIds(imgIds=self.imgIds[idx], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        w_img, h_img = size

        for ann in anns:
            label = self.coco.loadCats([ann['category_id']])[0]['name']
            x, y, w, h = ann['bbox']
            x, y = float(x)/w_img, float(y)/h_img
            w, h = float(w)/w_img, float(h)/h_img

            x_center, y_center = x + w / 2, y + h / 2

            num_label = self.label_dict[label]
            labels.append(num_label)
            bboxes.append((x_center, y_center, w, h))
        return labels, bboxes

    def make_target(self, labels, bboxes):
        """make location np.ndarray from bboxes of an image
        
        Parameters
        ----------
        labels : list
            [0, 1, 4, 2, ...]
            labels of each bboxes
        bboxes : list
            [[x_center, y_center, width, height], ...]
        
        Returns
        -------
        np.ndarray
            [self.S, self.S, self.B*5+self.C]
            location array
        """

        num_elements = self.B*5 + self.C
        num_bboxes = len(bboxes)
        
        # for excetion: num of bboxes is zero
        if num_bboxes == 0:
            return np.zeros((self.S, self.S, num_elements))

        labels = np.array(labels, dtype=np.int)
        bboxes = np.array(bboxes, dtype=np.float)

        np_target = np.zeros((self.S, self.S, num_elements))
        np_class = np.zeros((num_bboxes, self.C))

        for i in range(num_bboxes):
            np_class[i, labels[i]] = 1

        x_center = bboxes[:, 0].reshape(-1, 1)
        y_center = bboxes[:, 1].reshape(-1, 1)
        w = bboxes[:, 2].reshape(-1, 1)
        h = bboxes[:, 3].reshape(-1, 1)

        x_idx = np.ceil(x_center * self.S) - 1
        y_idx = np.ceil(y_center * self.S) - 1
        # for exception 0, ceil(0)-1 = -1
        x_idx[x_idx<0] = 0
        y_idx[y_idx<0] = 0

        # calc offset of x_center, y_center
        x_center = x_center - x_idx/self.S - 1/(2*self.S)
        y_center = y_center - y_idx/self.S - 1/(2*self.S)

        conf = np.ones_like(x_center)

        temp = np.concatenate([x_center, y_center, w, h, conf], axis=1)
        temp = np.repeat(temp, self.B, axis=0).reshape(num_bboxes, -1)
        temp = np.concatenate([temp, np_class], axis=1)

        for i in range(num_bboxes):
            np_target[int(y_idx[i]), int(x_idx[i])] = temp[i]

        return np_target

if __name__ == '__main__':
    # ms coco
    import pdb
    import matplotlib.pyplot as plt
    from utils import make_label_dict, visualize
    label_dict = make_label_dict('dataloader/coco_classes_names.txt')
    dataset = COCODataset(S=7, B=2, C=20,
                          label_dict=label_dict,
                          img_root='/data/coco/2017/val2017/',
                          ann_path='/data/coco/2017/annotations/instances_val2017.json')
    for i in range(len(dataset)):
        print(i)
        dataset[i]