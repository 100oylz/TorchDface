import torch
import torch.nn as nn
import torch.nn.functional as f
from network import weights_init


class Onet(nn.Module):
	def __init__(self, is_train=False, use_cuda=False):
		super(Onet, self).__init__()
		self.is_train = is_train
		self.use_cuda = use_cuda
		self.pre_layer=nn.Sequential(
			nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0),
			nn.RReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
			nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=0),
			nn.RReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2),
			nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=0),
			nn.RReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2),
			nn.RReLU()
		)
		self.conv1=nn.Linear(3*3*128,256)
		self.rrelu=nn.RReLU()
		self.conv1_1_2=nn.Linear(256,1)
		self.conv1_1_4 = nn.Linear(256, 4)
		self.conv1_1_10 = nn.Linear(256, 10)
		self.apply(weights_init.weights_init)
	def forward(self,x):
		x=self.pre_layer(x)
		x = x.view(x.size(0), -1)
		x=self.conv1(x)
		self.rrelu(x)
		label=torch.sigmoid(self.conv1_1_2(x))
		offset=self.conv1_1_4(x)
		landmark=self.conv1_1_10(x)
		return label,offset,landmark
	

