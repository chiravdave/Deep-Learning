import torch
from torch import nn


class InceptionBlockV1(nn.Module):
	def __init__(self, in_channels, filters_1x1, filters_3x3_reduce, 
		filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
		super(InceptionBlockV1, self).__init__()
		# 1X1 Conv
		self.conv1x1 = nn.Conv2d(
			in_channels, filters_1x1, 1, stride=1, padding="valid")
		self.act1x1 = nn.ReLU()

		# 3X3 Conv
		self.conv3x3_reduce = nn.Conv2d(
			in_channels, filters_3x3_reduce, 1, stride=1, padding="valid")
		self.act3x3_reduce = nn.ReLU()
		self.conv3x3 = nn.Conv2d(
			filters_3x3_reduce, filters_3x3, 3, stride=1, padding=1)
		self.act3x3 = nn.ReLU()

		# 5x5 Conv
		self.conv5x5_reduce = nn.Conv2d(
			in_channels, filters_5x5_reduce, 1, stride=1, padding="valid")
		self.act5x5_reduce = nn.ReLU()
		self.conv5x5 = nn.Conv2d(
			filters_5x5_reduce, filters_5x5, 5, stride=1, padding=2)
		self.act5x5 = nn.ReLU()

		# Max Pool
		self.pool = nn.MaxPool2d(3, stride=1, padding=1)
		self.pool_proj = nn.Conv2d(
			in_channels, filters_pool_proj, 1, stride=1, padding="valid")
		self.act_pool_proj = nn.ReLU()

	def forward(self, x):
		# 1x1 Conv Output
		out_1x1 = self.act1x1(self.conv1x1(x))

		# 3x3 Conv Output
		out_3x3_reduce = self.act3x3_reduce(self.conv3x3_reduce(x))
		out_3x3 = self.act3x3(self.conv3x3(out_3x3_reduce))

		# 5x5 Conv Output
		out_5x5_reduce = self.act5x5_reduce(self.conv5x5_reduce(x))
		out_5x5 = self.act5x5(self.conv5x5(out_5x5_reduce))

		# Max Pool Output
		out_pool = self.act_pool_proj(self.pool_proj(self.pool(x)))

		return torch.cat((out_1x1, out_3x3, out_5x5, out_pool), dim=1)

class InceptionV1(nn.Module):
	def __init__(self):
		self.batch_size = 128
		super(InceptionV1, self).__init__()
		self.layers = nn.ModuleList([
			nn.Conv2d(3, 64, 7, stride=2, padding=3),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1),
			nn.Conv2d(64, 64, 1, stride=1, padding="valid"),
			nn.ReLU(),
			nn.Conv2d(64, 192, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1)
		])
		# Inception block 3a
		self.layers.append(InceptionBlockV1(192, 64, 96, 128, 16, 32, 32))
		# Inception block 3b
		self.layers.append(InceptionBlockV1(256, 128, 128, 192, 32, 96, 64))
		self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
		# Inception block 4a
		self.layers.append(InceptionBlockV1(480, 192, 96, 208, 16, 48, 64))
		# Inception block 4b
		self.layers.append(InceptionBlockV1(512, 160, 112, 224, 24, 64, 64))
		# Inception block 4c
		self.layers.append(InceptionBlockV1(512, 128, 128, 256, 24, 64, 64))
		# Inception block 4d
		self.layers.append(InceptionBlockV1(512, 112, 144, 288, 32, 64, 64))
		# Inception block 4e
		self.layers.append(InceptionBlockV1(528, 256, 160, 320, 32, 128, 128))
		self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
		# Inception block 5a
		self.layers.append(InceptionBlockV1(832, 256, 160, 320, 32, 128, 128))
		# Inception block 5b
		self.layers.append(InceptionBlockV1(832, 384, 192, 384, 48, 128, 128))
		# FC layers
		self.layers.extend([
			nn.AvgPool2d(7, stride=1),
			nn.Dropout(p=0.4),
			nn.Linear(1024, 1000)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

if __name__ == "__main__":
	inceptionv1 = InceptionV1()
	print(inceptionv1)