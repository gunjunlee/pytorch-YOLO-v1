import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms

import random

from models import Model, YOLO
from loss import Loss
from metric import Metric
from dataloader import YOLODataset, BaseDataLoader
from utils import make_label_dict

S, B, C = 7, 2, None

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/train', help='path to training data')
    add_arg('--batch_size', default=32, type=int, help='batch size')
    add_arg('--lr', default=1e-3, type=float, help='learning rate')
    add_arg('--epoch', default=100, type=int, help='num of epochs')
    add_arg('--gpu', default=True, type=bool, help='use gpu')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    random.seed(1000)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    to_tensor = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    label_dict = make_label_dict('data')
    C = len(label_dict)
    dataset = YOLODataset('data/train', S, B, C,
                          label_dict=label_dict,
                          to_tensor=to_tensor)
    dataloader = BaseDataLoader(dataset=dataset, batch_size=args.batch_size)

    len_train = len(dataset) * 8 // 10
    len_val = len(dataset) - len_train
    train_dataloader, val_dataloader = dataloader.split([len_train, len_val])

    net = YOLO(S, B, C)
    if args.gpu:
        net = net.cuda()
    net = nn.DataParallel(net)

    criterion = Loss(S, B, C)
    metric = Loss(S, B, C)
    optimizer = optim.SGD(filter(lambda p:p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    model = Model(net)

    model.compile(optimizer, criterion, metric, scheduler, label_dict)
    model.fit(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epoch=args.epoch, use_gpu=args.gpu)