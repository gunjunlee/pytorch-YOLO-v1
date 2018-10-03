# Pytorch Implementation of YOLO v1

----

# NOT COMPLETED YET!

**Recommended directory structure**
```dir structure
- data
  - train
    - images
      - [image name].[png/jpg/bmp]
      - ...
    - bboxes
      - [image name].txt
      - ...
  - test
    - images
      - ...
    - bboxes
      - ...
  - class_names.txt
- dataloader
  - dataloader.py
- loss
  - loss.py
- metric
  - metric.py
- models
  - model.py
  - yolo.py
- test.py
- train.py
```

**class_names.txt Format**
```
{1st-label}
...
{nth-label}

ex) class_names.txt:
cat
dog
...
mouse
```

**BBox Format**
```bbox format
shape of bbox:
(x0, y0)-----(x1, y0)
   |            |
   |            |
(x0, y1)-----(x1, y1)
[x,y][0,1] is normalized coordinate.
ex) if size of image is (100, 100) and a point is located at (30, 40), then (x, y) of the point is (0.3, 0.4).

{image name}.txt:
{label} {x0} {y0} {x1} {y1}
...
{label} {x0} {y0} {x1} {y1}

ex)
img0.txt:
car 0.23 0.4 0.34 0.8
ball 0.1 0.1 0.3 0.2
```

## How to Run

training:
```
python train.py
```

test:
```
python test.py
```

## References
[YOLO-v1 paper](https://arxiv.org/abs/1506.02640)