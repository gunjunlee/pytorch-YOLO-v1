import numpy as np
from PIL import ImageDraw

import os
import os.path as osp

def make_label_dict(path):
    """make label dictionary
    
    Parameters
    ----------
    path : str
        path to class names file
    
    Returns
    -------
    dict
        {'person': 0, 'car': 1, ...}
    """

    label_dict = dict()
    with open(osp.join(path), 'r') as f:
        for class_name in f:
            label_dict[class_name.strip()] = len(label_dict)
    return label_dict

def visualize(img, pred, label_dict, thres=0.5):
    """visualizing prediction of an image
    
    Parameters
    ----------
    img : PIL.Image
        source image
    pred : torch.tensor
        [S, S, B*5+C]
    label_dict: list
        {str: int, ...}
    thres: float
        threshold of confidence
    """
    labelnum2name = dict()
    for k, v in label_dict.items():
        labelnum2name[v] = k
    confident_bboxes = get_confident_bboxes(pred, len(label_dict), thres)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for confident_bbox in confident_bboxes:
        # print(confident_bbox)
        confident_bbox = np.array(confident_bbox) * np.array([w, h, w, h, 1, 1])
        label_name = labelnum2name[int(confident_bbox[5])]
        draw.rectangle(list(confident_bbox)[:4], outline='red')
        draw.text(list(confident_bbox)[:2], '{},{:.3f}'.format(label_name, confident_bbox[4]), (255, 255, 0))
    del draw
    return img

def get_confident_bboxes(pred, num_classes, thres=0.5):
    S, _, num_elements = pred.shape
    C = num_classes
    B = (num_elements - C) // 5
    coords = pred[:, :, :B*5]
    classes = pred[:, :, B*5:]
    
    confident_bboxes = []

    for i in range(S):
        for j in range(S):
            for k in range(0, B*5, 5):
                if(pred[i][j][k+4].item() > thres):
                    x_center = pred[i][j][k+0].item()
                    y_center = pred[i][j][k+1].item()
                    x_center += j/S + 1/(2*S)
                    y_center += i/S + 1/(2*S)
                    w = pred[i][j][k+2].item()
                    h = pred[i][j][k+3].item()
                    x0 = x_center - w/2
                    y0 = y_center - h/2
                    x1 = x_center + w/2
                    y1 = y_center + h/2
                    conf = pred[i][j][k+4].item()
                    label = np.argmax(pred[i][j][B*5:])
                    bbox = (x0, y0, x1, y1, conf, label)
                    confident_bboxes.append(bbox)
    
    return confident_bboxes