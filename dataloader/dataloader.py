import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision import transforms

from PIL import Image
import numpy as np

import os
import os.path as osp


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def split(self, lengths, shuffle=False):
        if sum(lengths) != self.__len__():
            ValueError("Sum of input lengths does not match the length of the input dataset!")

        from torch.utils.data import Subset

        def accumulate(iterable):
            it = iter(iterable)
            total = 0
            for element in it:
                total += element
                yield(total)

        indices = [i for i in range(sum(lengths))]

        if shuffle:
            import random
            random.shuffle(indices)

        return [Subset(self, indices[offset-length: offset]) for offset, length in zip(accumulate(lengths), lengths)]


class BaseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        """Base dataloader

        Parameters
        ----------
        dataset : BaseDataset
        batch_size : int
        shuffle: bool
        sampler: torch.utils.data.sampler
        batch_sampler: torch.utils.data.
        num_workers: int
        collate_fn: function
        pin_memory: bool
        drop_last: bool
        timeout: 0
        worker_init_fn: function
        """

        self.dataset = dataset
        self.kwargs = kwargs
        super(BaseDataLoader, self).__init__(dataset=dataset, **kwargs)

    def split(self, lengths, shuffle=False):
        """split dataloader

        Parameters
        ----------
        lengths : iterable, [int, int, ...]
            lengths of sub-dataloaders
        shuffle : bool, optional
            shuffle indices of data (the default is False, which do not shuffle indices)

        Returns
        -------
        iterable [BaseDataLoader, BaseDataLoader, ...]
            return splited dataloaders
            their share one dataset
        """

        return [BaseDataLoader(dataset=subset, **self.kwargs) for subset in self.dataset.split(lengths, shuffle=shuffle)]

class YOLODataset(BaseDataset):
    def __init__(self, path, S, B, C, label_dict, to_tensor):
        """[summary]
        
        Parameters
        ----------
        path : [type]
            [description]
        S : [type]
            [description]
        B : [type]
            [description]
        C : [type]
            [description]
        label_dict : [type]
            {'label': 0, 'label': 1, ...}
        
        """

        self.path = path
        self.S = S
        self.B = B
        self.C = C
        self.label_dict = label_dict
        self.to_tensor = to_tensor
        self.names = []

        for path, _, fnames in os.walk(osp.join(self.path, 'images')):
            for fname in fnames:
                self.names.append(fname)
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        img = self.get_image(idx)
        labels, bboxes = self.get_bboxes(idx)
        target = self.make_target(labels, bboxes)
        
        img = self.to_tensor(img)
        target = torch.tensor(target)

        return img, target

    def make_target(self, labels, bboxes):
        num_elements = self.B*5 + self.C
        num_bboxes = len(bboxes)

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

    def get_image(self, idx):
        img = Image.open(osp.join(self.path, 'images', self.names[idx]))
        img = img.convert('RGB')
        return img

    def get_bboxes(self, idx):
        bboxes = []
        labels = []
        name, _ = osp.splitext(self.names[idx])
        with open(osp.join(self.path, 'bboxes', name+'.txt'), 'r') as f:
            for line in f:
                label, x0, y0, x1, y1 = line.strip().split(' ')
                x0, y0 = float(x0), float(y0)
                x1, y1 = float(x1), float(y1)
                if not (0<=x0<=1 and 0<=y0<=1 and 0<=x1<=1 and 0<=y1<=1):
                    raise ValueError('coord is not in range [0,1]!')
                x_center, y_center = (x0 + x1) / 2, (y0 + y1) / 2
                width, height = x1-x0, y1-y0
                if width < 0:
                    raise ValueError('width of obj is negative!')
                if height < 0:
                    raise ValueError('height of obj is negative!')

                num_label = self.label_dict[label]
                labels.append(num_label)
                bboxes.append((x_center, y_center, width, height))
        return labels, bboxes

if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    import sys, pdb
    import matplotlib.pyplot as plt
    sys.path.append('.')
    from utils import make_label_dict, visualize
    
    label_dict = make_label_dict('./data')
    print(label_dict)
    dataset = YOLODataset('./data/train', 7, 2, len(label_dict), label_dict, to_tensor)
    img, target = dataset[0]
    img = visualize(Image.open('data/train/images/002.jpg'), target, dataset.C, thres=0.5)
    plt.imshow(img)
    plt.show(img)