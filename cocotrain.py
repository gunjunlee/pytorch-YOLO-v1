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
from dataloader import COCODataset, BaseDataLoader
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

    label_dict = make_label_dict('dataloader/coco_classes_names.txt')
    C = len(label_dict)

    val_dataset = COCODataset(S=S, B=B, C=C,
                              label_dict=label_dict,
                              img_root='/data/coco/2017/val2017/',
                              ann_path='/data/coco/2017/annotations/instances_val2017.json')
    train_dataset = COCODataset(S=S, B=B, C=C,
                                label_dict=label_dict,
                                img_root='/data/coco/2017/train2017/',
                                ann_path='/data/coco/2017/annotations/instances_train2017.json')

    val_dataloader = BaseDataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8)
    train_dataloader = BaseDataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8)
    

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
    model.fit(train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              epoch=args.epoch,
              use_gpu=args.gpu)