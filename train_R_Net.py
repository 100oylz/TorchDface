import train
import os
from network import Rnet

batchsize = 16
max_epoch = 100
# max_epoch=10
data_path_relative = "./data/MTCNN/24"

if __name__ == '__main__':
    net = Rnet.Rnet()
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    Trainer = train.Trainer(net, './checkpoints/r_net_{0}.pt', data_path_relative, batchsize, max_epoch)
    Trainer.trainer()
