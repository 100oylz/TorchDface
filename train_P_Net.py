import train
import os
from network import PRO_Net

batchsize = 16
max_epoch = 100
# max_epoch=10
data_path_relative = "./data/MTCNN/12"

if __name__ == '__main__':
    net = PRO_Net.PNet()
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    Trainer = train.Trainer(net, './checkpoints/p_net_{0}.pt', data_path_relative, batchsize, max_epoch)
    Trainer.trainer()
