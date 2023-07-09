import train
import os
from network import Onet

batchsize = 32
max_epoch = 10
# max_epoch=10
data_path_relative = "./data/MTCNN/48"

if __name__ == '__main__':
    net = Onet.Onet()
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    Trainer = train.Trainer(net, './checkpoints/o_net_{0}.pt', data_path_relative, batchsize, max_epoch,netName='o_net')
    Trainer.trainer()
