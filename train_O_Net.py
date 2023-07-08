import train
import os
from network import PRO_Net

batchsize = 16
max_epoch = 100
# max_epoch=10
data_path_relative = "./data/MTCNN/48"

if __name__ == '__main__':
    net = PRO_Net.ONet()
    if not os.path.exists("../models"):
        os.makedirs("../models")
    Trainer = train.Trainer(net, '../models/o_net_{0}.pt', data_path_relative, batchsize, max_epoch,netName='o_net')
    Trainer.trainer()
