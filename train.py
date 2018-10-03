import torch
import torch.nn as nn

import random

from models import Model, YOLO
from loss import Loss
from metric import Metric
from dataloader import YOLODataset, BaseDataLoader
from utils import make_label_dict


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/train', help='path to training data')
    add_arg('--batch_size', default=32, type=int, help='batch size')
    add_arg('--lr', default=1e-3, type=float, help='learning rate')
    add_arg('--gpu', default=True, type=bool, help='use gpu')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    random.seed(1000)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    label_dict = make_label_dict('data')
    dataset = YOLODataset('data/train', 7, 2, len(label_dict),
                          label_dict=label_dict,
                          to_tensor=to_tensor)
    dataloader = BaseDataLoader(dataset=dataset, batch_size=args.batch_size)

    len_train = len(dataset) * 8 // 10
    len_val = len(dataset) - len_train
    train_dataloader, val_dataloader = dataloader.split([len_train, len_val])

    net = YOLO(7, 2, len(label_dict))
    if args.gpu:
        net = net.cuda()
    net = nn.DataParallel(net)

    criterion
    optimizer
    scheduler
    
    model = Model(net)

    model.compile()
    model.fit()