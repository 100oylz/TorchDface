import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = []
        for fi in ['negative.txt', 'positive.txt', 'part.txt']:
            l = open(os.path.join(data_path, fi)).readlines()
            for l_filename in l:
                self.dataset.append(l_filename.split(" ")[:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img_tensor = self.trans(Image.open(data[0]))
        category = torch.tensor(float(data[1])).reshape(-1)
        bbox_offset = torch.tensor([float(data[2]), float(data[3]), float(data[4]), float(data[5])])
        landmark_offset = torch.tensor([float(data[6]), float(data[7]), float(data[8]), float(data[9]),
                                        float(data[10]), float(data[11]), float(data[12]), float(data[13]),
                                        float(data[14]), float(data[15])])

        return img_tensor, category, bbox_offset, landmark_offset

    def trans(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ], [0.5, ])
        ])(x)


class Trainer:
    def __init__(self, net, save_path, dataset_path=None, batchsize=1, max_epoch=100, netName=None):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)  # 通用的属性加self
        self.batchsize = batchsize
        self.cur_epoch = 0
        self.max_epoch = max_epoch
        self.save_path = save_path
        self.netName = netName
        self.dataset_path = dataset_path
        self.faceDataset = FaceDataset(self.dataset_path)  # 实例化对象
        self.dataloader = DataLoader(self.faceDataset, self.batchsize, shuffle=True, num_workers=1, drop_last=True)
        self.cls_loss_fn = nn.BCELoss()  # 置信度损失函数
        self.offset_loss_fn = nn.MSELoss()  # bbx坐标偏移量损失函数
        # self.offset_loss_fn = nn.SmoothL1Loss()  # bbx坐标偏移量损失函数
        self.landmark_offset_loss_fn = nn.MSELoss()  # landmarks偏移量损失函数
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
        # if epoch < 50:
        #     self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.9)
        # else:
        #     self.optimizer = optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
        #     阶梯下降的学习率优化策略
            # lr = lr*0.9

        # if os.path.exists(self.save_path):  # 是否有已经保存的参数文件
        #     net.load_state_dict(torch.load(self.save_path, map_location=self.device))

    def trainer(self):
        loss, cls_loss, bbox_offset_loss, landmark_offset_loss = 0, 0, 0, 0
        self.net.train()
        # self.net.eval()
        while self.cur_epoch <= self.max_epoch:
            cla_label = []
            cla_out = []
            bbox_offset_label = []
            bbox_offset_out = []
            landmark_offset_label = []
            landmark_offset_out = []
            for i, (img_data_, category_gt, bbox_offset_gt, landmark_offset_gt) in enumerate(self.dataloader):
                img_data_ = img_data_.to(self.device)  # 得到的三个值传入到CPU或者GPU
                category_gt = category_gt.to(self.device)
                bbox_offset_gt = bbox_offset_gt.to(self.device)
                landmark_offset_gt = landmark_offset_gt.to(self.device)

                _category_pr, _bbox_offset_pr, _landmark_offset_pr = self.net(img_data_)  # 输出置信度和偏移值
                category_pr = _category_pr.view(-1, 1)  # 转化成NV结构
                bbox_offset_pr = _bbox_offset_pr.view(-1, 4)
                landmark_offset_pr = _landmark_offset_pr.view(-1, 10)
                # print(category_pr.shape)
                # print(output_offset.shape, "=================")

                # 正样本和负样本用来训练置信度

                '''
                category_gt,bbox_offset_gt,landmark_offset_gt是数据集样本的取值
                category_pr,bbox_offset_pr,landmark_offset_pr是数据集样本的取值

                获得小于2的人脸分类标签,即正样本和负样本,因为我们在设置标签的时候指定0为负样本，1为正样本，2为部分正样本
                '''
                # 计算分类损失
                # 小于2   #一系列布尔值  逐元素比较input和other ， 即是否 \( input < other \)，第二个参数可以为一个数或与第一个参数相同形状和类型的张量。
                category_mask = torch.lt(category_gt, 2)
                category = torch.masked_select(category_gt, category_mask)  # 取到对应位置上的标签置信度
                # torch.masked_select()根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，
                # 上面两行等价于category_mask = category[category < 2]
                category_pr = torch.masked_select(category_pr, category_mask)  # 输出的置信度
                cls_loss = self.cls_loss_fn(category_pr, category)  # 计算置信度的损失

                '''
                获得大于0的人脸分类标签,即部分样本和正样本
                '''
                # 计算bbx回归损失
                bbox_offset_mask = torch.gt(category_gt, 0)  # torch.gt(a,b)函数比较a中元素大于（这里是严格大于）b中对应元素，大于则为1，不大于则为0，
                bbx_offset = torch.masked_select(bbox_offset_gt, bbox_offset_mask)
                bbox_offset_pr = torch.masked_select(bbox_offset_pr, bbox_offset_mask)
                bbox_offset_loss = self.offset_loss_fn(bbox_offset_pr, bbx_offset)  # 计算偏移值的损失

                # 计算landmark回归损失
                landmark_offset_mask = torch.gt(category_gt, 0)
                landmark_offset = torch.masked_select(landmark_offset_gt, landmark_offset_mask)
                landmark_offset_gt = torch.masked_select(landmark_offset_pr, landmark_offset_mask)
                landmark_offset_loss = self.landmark_offset_loss_fn(landmark_offset_gt, landmark_offset)  # 计算偏移值的损失

                # P、R网络对landmark要求不高，主要是过滤bbx，因此不添加landmark_loss进去,
                # 尽可能的减小landmark造成的损失过大影响，O网络再加进去
                # landmark在P、R网络内不贡献梯度，也就不参与w，b的更新，
                # 因为landmark前期预测大概率不准，对模型收敛有较大影响
                if self.netName == 'o_net':
                    loss = cls_loss + bbox_offset_loss + 2*landmark_offset_loss
                else:
                    loss = cls_loss + bbox_offset_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()  # 更新梯度反向传播

                cls_loss = cls_loss.cpu().item()  # 将损失转达CPU上计算，此处的损失指的是每一批次的损失
                bbox_offset_loss = bbox_offset_loss.cpu().item()
                landmark_offset_loss = landmark_offset_loss.cpu().item()
                loss = loss.cpu().item()

                cla_out.extend(category_pr.detach().cpu())
                cla_label.extend(category.detach().cpu())
                bbox_offset_out.extend(bbox_offset_pr.detach().cpu())
                bbox_offset_label.extend(bbx_offset.detach().cpu())
                landmark_offset_out.extend(bbox_offset_pr.detach().cpu())
                landmark_offset_label.extend(bbx_offset.detach().cpu())

                cla_out = []
                cla_label.clear()
                bbox_offset_out.clear()
                bbox_offset_label.clear()
                landmark_offset_out.clear()
                landmark_offset_label.clear()
            # o_net计算总损失=分类损失+bbox回归损失+landmark回归损失，p_net r_net总损失=分类损失+bbox回归损失
            if self.netName == 'o_net':
                print("epoch:%d loss:%.4f cls_loss:%.4f bbox_offset_loss:%.4f landmark_offset_loss:%.4f" % (
                    self.cur_epoch, loss, cls_loss, bbox_offset_loss, landmark_offset_loss))
            else:
                print("epoch:%d loss:%.4f cls_loss:%.4f bbox_offset_loss:%.4f" % (
                    self.cur_epoch, loss, cls_loss, bbox_offset_loss))

            if self.cur_epoch % 50 == 0 and self.cur_epoch != 0:  # 策略1：每50个epoch保存一次，策略2：loss总损失永远保存最小的那一次
                torch.save(self.net.state_dict(), self.save_path.format(self.cur_epoch))
                # 保存模型的推理过程的时候，只需要保存模型训练好的参数，
                # 使用torch.save()保存state_dict，能够方便模型的加载
                print("save success")
            self.cur_epoch += 1

        print("train finished!!!")
