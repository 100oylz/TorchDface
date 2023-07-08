# -*- coding: utf-8 -*-
import torch
from PIL import Image
import numpy as np
import os
import cv2
from torchvision import transforms
import utils
from network import PRO_Net

p_net_cls = 0.2
r_net_cls = 0.6


class Detector:
    def __init__(self):
        # self.device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cpu"
        self.pnet = PRO_Net.PNet()
        self.rnet = PRO_Net.RNet()
        self.onet = PRO_Net.ONet()

        self.pnet.to(self.device)
        self.rnet.to(self.device)
        self.onet.to(self.device)

        self.pnet.load_state_dict(torch.load("checkpoints/p_net_100.pt", map_location=self.device))
        self.rnet.load_state_dict(torch.load("checkpoints/r_net_100.pt", map_location=self.device))
        self.onet.load_state_dict(torch.load("checkpoints/o_net_100.pt", map_location=self.device))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        # 记住一定要使用model.eval()来固定dropout和归一化层，否则每次推理会生成不同的结果

        self._image_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transforms.Normalize([0.5, ], [0.5, ], [0.5, ])
        ])

    def detect(self, image):
        pnet_boxes = self._pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        rnet_boxes = self._rnet_detect(image, pnet_boxes)  # p网络输出的框和原图像输送到R网络中，O网络将框扩为正方形再进行裁剪，再缩放
        if rnet_boxes.shape[0] == 0:
            return pnet_boxes, np.array([]), np.array([])

        onet_boxes = self._onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return pnet_boxes, rnet_boxes, np.array([])

        return pnet_boxes, rnet_boxes, onet_boxes

    def _pnet_detect(self, img):

        boxes = []
        w, h = img.size
        min_side_len = min(w, h)
        # min_side_len = max(w, h)

        scale = 1

        # 图象金字塔
        while min_side_len >= 12:
            img_data = self._image_transform(img)
            img_data = img_data.to(self.device)
            img_data.unsqueeze_(0)  # 升维度（新版pytorch可以删掉）

            _cls, _bbox_offest, _landmark_offset = self.pnet(img_data)  # NCHW
            cls = _cls[0][0].cpu().data
            bbox_offest = _bbox_offest[0].cpu().data
            landmark_offset = _landmark_offset[0].cpu().data
            # 找出非0元素索引（置信度严格大于p_net_cls=0.2替换成1，反之替换成0）,做初步过滤用
            idxs = torch.nonzero(torch.gt(cls, p_net_cls), as_tuple=False)

            for idx in idxs:  # idx里面就是一个h和一个w
                boxes.append(self._box(idx, bbox_offest, cls[idx[0], idx[1]], scale))  # 把对应位置取出来，进行反向映射到原图上
            scale *= 0.709  # 缩放因子
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)
        return utils.nms(np.array(boxes), 0.3)

    def _box(self, start_index, offset, cls, scale, stride=2, side_len=12):  # side_len=12建议框大大小

        _x1 = int(start_index[1] * stride) / scale  # 宽，W，x
        _y1 = int(start_index[0] * stride) / scale  # 高，H,y
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1  # 偏移量
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]  # 通道层面全都要[C, H, W]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def _rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            if(_x1>_x2) or (_y1>_y2):
                continue
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self._image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _bbox_offset, _landmark_offset = self.rnet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        bbox_offset = _bbox_offset.cpu().data.numpy()
        # landmark_offset = _landmark_offset.cpu().data.numpy()
        boxes = []

        idxs, _ = np.where(_cls > r_net_cls)
        for idx in idxs:  # 只是取出合格的
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * bbox_offset[idx][0]
            y1 = _y1 + oh * bbox_offset[idx][1]
            x2 = _x2 + ow * bbox_offset[idx][2]
            y2 = _y2 + oh * bbox_offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])
        return utils.nms(np.array(boxes), 0.3)

    def _onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self._image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _bbox_offset, _landmark_offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        bbox_offset = _bbox_offset.cpu().data.numpy()
        landmark_offset = _landmark_offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(_cls > 0.90)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1
            # bbox坐标还原
            x1 = _x1 + ow * bbox_offset[idx][0]
            y1 = _y1 + oh * bbox_offset[idx][1]
            x2 = _x2 + ow * bbox_offset[idx][2]
            y2 = _y2 + oh * bbox_offset[idx][3]
            # 置信度
            cls = _cls[idx][0]
            # landmark坐标还原
            side_len = max(ow, oh)
            x11 = _x1 + side_len * landmark_offset[idx][0]
            y11 = _y1 + side_len * landmark_offset[idx][1]
            x22 = _x1 + side_len * landmark_offset[idx][2]
            y22 = _y1 + side_len * landmark_offset[idx][3]
            x33 = _x1 + side_len * landmark_offset[idx][4]
            y33 = _y1 + side_len * landmark_offset[idx][5]
            x44 = _x1 + side_len * landmark_offset[idx][6]
            y44 = _y1 + side_len * landmark_offset[idx][7]
            x55 = _x1 + side_len * landmark_offset[idx][8]
            y55 = _y1 + side_len * landmark_offset[idx][9]

            boxes.append([x1, y1, x2, y2, cls, x11, y11, x22, y22, x33, y33, x44, y44, x55, y55])

        return utils.nms(np.array(boxes), 0.1)


def test_pic():
    # path = "D:\软件\BaiduNetdisk\百度网盘下载\CelebA\Img\img_align_celeba"
    path = ".\evaluate\eval_pictures"
    # path = "D:\workspace\project\IMAGE\Dface\evaluate\pictures"

    for pic_name in os.listdir(path):
        # img = cv2.imread(os.path.join(path, pic_name))
        # 如果使用中文路径会出错，没有中文路径使用imread方法即可
        img = cv2.imdecode(np.fromfile(os.path.join(path, pic_name), dtype=np.uint8), -1)
        im = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        detector = Detector()
        pnet_boxes, rnet_boxes, onet_boxes = detector.detect(im)
        if onet_boxes == np.array([]):
            continue
        # 这里我们只需要o_net的输出结果就行，p和r的都不看，需要看的遍历zip(pnet_boxes, rnet_boxes, onet_boxes)
        for box in onet_boxes:  # 训练的不好，有很多无用框，直接取最好的
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            conf = box[4]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img, str(round(conf, 3)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                        lineType=cv2.LINE_AA)
            # 画五官landmark点
            x11, y11 = int(box[5]), int(box[6])
            x22, y22 = int(box[7]), int(box[8])
            x33, y33 = int(box[9]), int(box[10])
            x44, y44 = int(box[11]), int(box[12])
            x55, y55 = int(box[13]), int(box[14])
            cv2.circle(img, (x11, y11), radius=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, (x22, y22), radius=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, (x33, y33), radius=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, (x44, y44), radius=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, (x55, y55), radius=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('pic', img)
        cv2.waitKey(0)


def test_video():
    # cap = cv2.VideoCapture('./evaluate/eval_vedios/660C2E29AFB92EE3DC62DF5D5B0B9301.mp4')
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('./evaluate/eval_vedios/VID_20210710_155549.mp4')
    # cap.set(4, 300)
    # cap.set(3, 300)
    # fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    # w = int(cap.get(3))
    # h = int(cap.get(4))

    while True:
        ret, img = cap.read()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif ret == False:
            break

        im = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        detector = Detector()
        _, _, boxes = detector.detect(im)
        if boxes is None:
            continue
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cls = box[4]

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            if len(box) > 5:
                x11, y11 = int(box[5]), int(box[6])
                x22, y22 = int(box[7]), int(box[8])
                x33, y33 = int(box[9]), int(box[10])
                x44, y44 = int(box[11]), int(box[12])
                x55, y55 = int(box[13]), int(box[14])
                cv2.circle(img, (x11, y11), radius=3, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(img, (x22, y22), radius=3, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(img, (x33, y33), radius=3, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(img, (x44, y44), radius=3, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(img, (x55, y55), radius=3, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            # cv2.rectangle(im, (x1, y1,), (x2, y2), (255, 0, 0), 3)
            # im = np.array(im).astype('uint8')
            # im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        cv2.imshow("video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    test_pic()
    # test_video()
