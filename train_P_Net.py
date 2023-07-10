import train
import os
from network import Pnet

batchsize = 16
max_epoch = 200
# max_epoch=10
data_path_relative = "./data/MTCNN/12"

if __name__ == '__main__':
    net = Pnet.Pnet()
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    Trainer = train.Trainer(net, './checkpoints/p_net_{0}.pt',  data_path_relative, batchsize, max_epoch)
    Trainer.trainer()
