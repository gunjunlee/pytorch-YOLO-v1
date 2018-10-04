import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

class Loss(nn.Module):
    def __init__(self, S, B, C, lambda_coord=5, lambda_noobj=0.5):
        super(Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def calc_iou(self, A, B):
        """calc iou A & B
        
        Args:
            A (torch.FloatTensor): [N, SxSx(Bx5+C)]
            B (torch.FloatTensor): [N, SxSx(Bx5+C)]
        """

        A = A.view(-1, self.S, self.S, self.B * 5 + self.C)
        B = B.view(-1, self.S, self.S, self.B * 5 + self.C)
        
        A_x_center = A[:, :, :, 0:self.B*5:5]
        A_y_center = A[:, :, :, 1:self.B*5:5]
        A_w = A[:, :, :, 2:self.B*5:5]
        A_h = A[:, :, :, 3:self.B*5:5]
        
        B_x_center = B[:, :, :, 0:self.B*5:5]
        B_y_center = B[:, :, :, 1:self.B*5:5]
        B_w = B[:, :, :, 2:self.B*5:5]
        B_h = B[:, :, :, 3:self.B*5:5]
        
        A_area = A_w * A_h
        B_area = B_w * B_h

        inter_box_x0, _ = torch.max(torch.cat([(A_x_center-A_w/2).unsqueeze(dim=-1), (B_x_center-B_w/2).unsqueeze(dim=-1)], dim=-1), dim=-1)
        inter_box_y0, _ = torch.max(torch.cat([(A_y_center-A_h/2).unsqueeze(dim=-1), (B_y_center-B_h/2).unsqueeze(dim=-1)], dim=-1), dim=-1)

        inter_box_x1, _ = torch.min(torch.cat([(A_x_center+A_w/2).unsqueeze(dim=-1), (B_x_center+B_w/2).unsqueeze(dim=-1)], dim=-1), dim=-1)
        inter_box_y1, _ = torch.min(torch.cat([(A_y_center+A_h/2).unsqueeze(dim=-1), (B_y_center+B_h/2).unsqueeze(dim=-1)], dim=-1), dim=-1)
        
        inter_box_w = inter_box_x1-inter_box_x0
        inter_box_h = inter_box_y1-inter_box_y0

        inter = inter_box_w * inter_box_h * (inter_box_h>0).float() * (inter_box_w>0).float()

        iou = inter / (A_area + B_area - inter + 1e-6)

        return iou
        
    def get_argmax_iou(self, A, B):
        """get argmax of iou A & B
        
        Args:
            A (torch.FloatTensor): [N, SxSx(Bx5+C)]
            B (torch.FloatTensor): [N, SxSx(Bx5+C)]
        """

        iou = self.calc_iou(A, B)
        """iou: [N, S, S, B]
        """
        
        argmax = torch.argmax(iou, dim=-1)
        return argmax

    def forward(self, pred, target):
        """calc loss
        
        Args:
            pred (torch.floatTensor): [N, S, S, (Bx5+C)]
            target (torch.floatTensor): [N, S, S, Bx5+C] score is always equal to 1. bbox: [x_center, y_center, w, h]
        """
        num_elements = self.B * 5 + self.C
        num_batch = target.size(0)
        
        target = target.view(-1, self.S*self.S, num_elements)
        pred = pred.view(-1, self.S*self.S, num_elements)
        """now target and pred: [N, SxS, (Bx5+C)]
        """

        obj_mask = target[:,:,4] > 0
        noobj_mask = target[:,:,4] == 0

        obj_mask = obj_mask.unsqueeze(-1).expand_as(target).float()
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target).float()
        """now obj_mask and noobj: [N, SxS, (Bx5+C)]
        """
        
        responsible_bbox_arg = self.get_argmax_iou(pred, target)
        responsible_bbox_scatter = torch.tensor((0, 1, 2, 3, 4))\
                                .repeat((num_batch, self.S * self.S, 1)).cuda()\
                                + responsible_bbox_arg.view(-1, self.S*self.S, 1)
        responsible_bbox_mask = torch.zeros((num_batch, self.S * self.S, self.B * 5 + self.C)).cuda()\
                                .scatter_(2, responsible_bbox_scatter, torch.ones((num_batch, self.S * self.S, self.B * 5 + self.C)).cuda())
        responsible_bbox_mask = responsible_bbox_mask * obj_mask

        # class prediction loss
        class_prediction_loss = ((torch.sigmoid(pred) - torch.sigmoid(target)) * obj_mask)[:, :, self.B*5:].pow(2).sum()

        # no obj loss
        noobj_loss = self.lambda_noobj * ((torch.sigmoid(pred) - torch.sigmoid(target)) * noobj_mask)[:, :, 4:self.B*5:5].pow(2).sum()

        # obj loss
        obj_loss = ((torch.sigmoid(pred) - torch.sigmoid(target)) * responsible_bbox_mask)[:, :, 4:self.B*5:5].pow(2).sum()

        # coord loss
        coord_xy_loss = self.lambda_coord * ((pred-target) * responsible_bbox_mask)[:, :, 0:self.B*5:5].pow(2).sum()\
                        + self.lambda_coord * ((pred-target) * responsible_bbox_mask)[:, :, 1:self.B*5:5].pow(2).sum()

        coord_wh_loss = self.lambda_coord * ((pred-target) * responsible_bbox_mask)[:, :, 2:self.B*5:5].pow(2).sum()\
                        + self.lambda_coord * ((pred-target) * responsible_bbox_mask)[:, :, 3:self.B*5:5].pow(2).sum()
        
        total_loss = class_prediction_loss + noobj_loss + obj_loss + coord_xy_loss + coord_wh_loss

        return total_loss/num_batch


if __name__ == '__main__':
    loss = YOLOLoss(1, 2, 2)
    pred = torch.tensor([[0.5, 0.5, 1, 0.5, 1, 0.5, 0.5, 1, 1, 1, 0, 1]]).float()
    target = torch.tensor([[0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 0, 1]]).float()
    loss.forward(pred, target)