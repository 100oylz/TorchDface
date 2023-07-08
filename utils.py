import numpy as np


def iou(box, boxes, Union=True):  # 计算选框box 与  选框列表 boxes的iou，
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h  # 重合部分的面积
    if Union:
        ovr = np.true_divide(inter, (box_area + area - inter))  # 正常的交并比
    else:
        ovr = np.true_divide(inter, np.minimum(box_area, area))  # 真正除法的运算结果     #大框套小框

    return ovr  # 返回结果


def nms(boxes, thresh=0.3, Union=True):
    if boxes.shape[0] == 0:  # 防止程序出错
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]  # conf从大到小排序
    r_boxes = []

    '''
    _boxes是排序好的选框
    r_boxes是最终输出的选框
    '''
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)
        index = np.where(iou(a_box, b_boxes, Union) < thresh)  # 返回满足条件的索引
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)


# https://blog.csdn.net/wgx571859177/article/details/80987459
# 对于axis = 1，就是横着切开，对应行横着堆
# 对于axis = 2，就是横着切开，对应行竖着堆
# 对于axis = 0，就是不切开，两个堆一起


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox
