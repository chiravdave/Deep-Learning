import torch
from torch import nn

CKPT_PATH = "model.pt"


class Conv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride=1, pad=0):
		super(Conv2D, self).__init__()
		self.layers = nn.ModuleList([
			nn.Conv2d(in_channels, out_channels, kernel, stride=stride, 
				padding=pad, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

class InceptionBlockA(nn.Module):
	def __init__(self, in_channels, filters_pool_proj):
		super(InceptionBlockA, self).__init__()
		# 1X1 Conv
		self.conv1x1 = Conv2D(in_channels, 64, 1)

		# Avg Pool
		self.pool = nn.AvgPool2d(3, stride=1, padding=1)
		self.pool_proj = Conv2D(in_channels, filters_pool_proj, 1)

		# 3x3 Conv
		self.conv3x3_reduce = Conv2D(in_channels, 64, 1)
		self.conv3x3_1 = Conv2D(64, 96, 3, pad=1)
		self.conv3x3_2 = Conv2D(96, 96, 3, pad=1)

		# 5x5 Conv
		self.conv5x5_reduce = Conv2D(in_channels, 48, 1)
		self.conv5x5 = Conv2D(48, 64, 5, pad=2)

	def forward(self, x):
		# 1x1 Conv Output
		out_1x1 = self.conv1x1(x)

		# Avg Pool Output
		out_pool = self.pool_proj(self.pool(x))

		# 3x3 Conv Output
		out_3x3_reduce = self.conv3x3_reduce(x)
		out_3x3_1 = self.conv3x3_1(out_3x3_reduce)
		out_3x3 = self.conv3x3_2(out_3x3_1)

		# 5x5 Conv Output
		out_5x5_reduce = self.conv5x5_reduce(x)
		out_5x5 = self.conv5x5(out_5x5_reduce)

		return torch.cat((out_1x1, out_3x3, out_5x5, out_pool), dim=1)

class InceptionBlockB(nn.Module):
	def __init__(self, in_channels):
		super(InceptionBlockB, self).__init__()
		# Max Pool
		self.pool = nn.MaxPool2d(3, stride=2)

		# 3x3 Conv Block-1
		self.conv3x3_b1 = Conv2D(in_channels, 384, 3, stride=2)

		# 3x3 Conv Block-2
		self.conv3x3_b2_reduce = Conv2D(in_channels, 64, 1)
		self.conv3x3_b2_1 = Conv2D(64, 96, 3, pad=1)
		self.conv3x3_b2_2 = Conv2D(96, 96, 3, stride=2)

	def forward(self, x):
		# Max Pool Output
		out_pool = self.pool(x)

		# 3x3 Conv Block-21 Output
		out_3x3_b1 = self.conv3x3_b1(x)

			# 3x3 Conv Block-2 Output
		out_3x3_b2_reduce = self.conv3x3_b2_reduce(x)
		out_3x3_b2_1 = self.conv3x3_b2_1(out_3x3_b2_reduce)
		out_3x3_b2 = self.conv3x3_b2_2(out_3x3_b2_1)

		return torch.cat((out_3x3_b1, out_3x3_b2, out_pool), dim=1)

class InceptionBlockC(nn.Module):
	def __init__(self, in_channels, channels_7x7):
		super(InceptionBlockC, self).__init__()
		# 1X1 Conv
		self.conv1x1 = Conv2D(in_channels, 192, 1)

		# Avg Pool
		self.pool = nn.AvgPool2d(3, stride=1, padding=1)
		self.pool_proj = Conv2D(in_channels, 192, 1)

		# 7x7 Conv Block-1
		self.conv7x7_b1_reduce = Conv2D(in_channels, channels_7x7, 1)
		self.conv7x7_b1_1 = Conv2D(
			channels_7x7, channels_7x7, (1, 7), pad=(0, 3))
		self.conv7x7_b1_2 = Conv2D(channels_7x7, 192, (7, 1), pad=(3, 0))

		# 7x7 Conv Block-2
		self.conv7x7_b2_reduce = Conv2D(in_channels, channels_7x7, 1)
		self.conv7x7_b2_1 = Conv2D(
			channels_7x7, channels_7x7, (1, 7), pad=(0, 3))
		self.conv7x7_b2_2 = Conv2D(
			channels_7x7, channels_7x7, (7, 1), pad=(3, 0))
		self.conv7x7_b2_3 = Conv2D(
			channels_7x7, channels_7x7, (1, 7), pad=(0, 3))
		self.conv7x7_b2_4 = Conv2D(channels_7x7, 192, (7, 1), pad=(3, 0))

	def forward(self, x):
		# 1x1 Conv Output
		out_1x1 = self.conv1x1(x)

		# Avg Pool Output
		out_pool = self.pool_proj(self.pool(x))

			# 7x7 Conv Block-1 Output
		out_7x7_b1_reduce = self.conv7x7_b1_reduce(x)
		out_7x7_b1 = self.conv7x7_b1_2(self.conv7x7_b1_1(out_7x7_b1_reduce))

			# 7x7 Conv Block-2 Output
		out_7x7_b2_reduce = self.conv7x7_b2_reduce(x)
		out_7x7_b2_o1 = self.conv7x7_b2_2(self.conv7x7_b2_1(out_7x7_b2_reduce))
		out_7x7_b2 = self.conv7x7_b2_4(self.conv7x7_b2_3(out_7x7_b2_o1))

		return torch.cat((out_1x1, out_7x7_b1, out_7x7_b2, out_pool), dim=1)

class InceptionBlockD(nn.Module):
	def __init__(self, in_channels):
		super(InceptionBlockD, self).__init__()
		# Max Pool
		self.pool = nn.MaxPool2d(3, stride=2)

		# 3x3 Conv
		self.conv3x3_reduce = Conv2D(in_channels, 192, 1)
		self.conv3x3 = Conv2D(192, 320, 3, stride=2)

		# 7x7 Conv
		self.conv7x7_reduce = Conv2D(in_channels, 192, 1)
		self.conv7x7_1 = Conv2D(192, 192, (1, 7), pad=(0, 3))
		self.conv7x7_2 = Conv2D(192, 192, (7, 1), pad=(3, 0))
		self.conv7x7_3 = Conv2D(192, 192, 3, stride=2)

	def forward(self, x):
		# Max Pool Output
		out_pool = self.pool(x)

		# 3x3 Conv Output
		out_3x3_reduce = self.conv3x3_reduce(x)
		out_3x3 = self.conv3x3(out_3x3_reduce)

		# 7x7 Output
		out_7x7_reduce = self.conv7x7_reduce(x)
		out_7x7_1 = self.conv7x7_1(out_7x7_reduce)
		out_7x7_2 = self.conv7x7_2(out_7x7_1)
		out_7x7 = self.conv7x7_3(out_7x7_2)

		return torch.concat((out_3x3, out7x7, out_pool), dim=1)

class InceptionBlockE(nn.Module):
	def __init__(self, in_channels):
		super(InceptionBlockE, self).__init__()
		# 1X1 Conv
		self.conv1x1 = Conv2D(in_channels, 320, 1)

		# Avg Pool
		self.pool = nn.AvgPool2d(3, stride=1, padding=1)
		self.pool_proj = Conv2D(in_channels, 192, 1)

		# 3x3 Conv Block-1
		self.conv3x3_b1_reduce = Conv2D(in_channels, 384, 1)
		self.conv3x3_b1_1 = Conv2D(384, 384, (1, 3), pad=(0, 1))
		self.conv3x3_b1_2 = Conv2D(384, 384, (3, 1), pad=(1, 0))

		# 3x3 Conv Block-2
		self.conv3x3_b2_reduce = Conv2D(in_channels, 448, 1)
		self.conv3x3_b2_1 = Conv2D(448, 384, 3, pad=1)
		self.conv3x3_b2_2 = Conv2D(384, 384, (1, 3), pad=(0, 1))
		self.conv3x3_b2_3 = Conv2D(384, 384, (3, 1), pad=(1, 0))

	def forward(self, x):
		# 1x1 Conv Output
		out_1x1 = self.conv1x1(x)

		# Avg Pool Output
		out_pool = self.pool_proj(self.pool(x))

			# 3x3 Conv Block-1 Output
		out_3x3_b1_reduce = self.conv3x3_b1_reduce(x)
		out_3x3_b1_1 = self.conv3x3_b1_1(out_3x3_b1_reduce)
		out_3x3_b1_2 = self.conv3x3_b1_2(out_3x3_b1_reduce)
		out_3x3_b1 = torch.cat((out_3x3_b1_1, out_3x3_b1_2), dim=1)

			# 3x3 Conv Block-2 Output
		out_3x3_b2_reduce = self.conv3x3_b2_reduce(x)
		out_3x3_b2_1 = self.conv3x3_b2_1(out_3x3_b2_reduce)
		out_3x3_b2_2 = self.conv3x3_b2_2(out_3x3_b2_1)
		out_3x3_b2_3 = self.conv3x3_b2_3(out_3x3_b2_1)
		out_3x3_b2 = torch.cat((out_3x3_b2_2, out_3x3_b2_3), dim=1)

		return torch.cat((out_1x1, out_3x3_b1, out_3x3_b2, out_pool), dim=1)

class InceptionBlockV1(nn.Module):
	def __init__(self, in_channels, filters_1x1, filters_3x3_reduce, 
		filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
		super(InceptionBlockV1, self).__init__()
		# 1X1 Conv
		self.conv1x1 = Conv2D(in_channels, filters_1x1, 1)

		# 3X3 Conv
		self.conv3x3_reduce = Conv2D(in_channels, filters_3x3_reduce, 1)
		self.conv3x3 = Conv2D(filters_3x3_reduce, filters_3x3, 3, pad=1)

		# 5x5 Conv
		self.conv5x5_reduce = Conv2D(in_channels, filters_5x5_reduce, 1)
		self.conv5x5 = Conv2D(filters_5x5_reduce, filters_5x5, 5, pad=2)

		# Max Pool
		self.pool = nn.MaxPool2d(3, stride=1, padding=1)
		self.pool_proj = Conv2D(in_channels, filters_pool_proj, 1)

	def forward(self, x):
		# 1x1 Conv Output
		out_1x1 = self.conv1x1(x)

		# 3x3 Conv Output
		out_3x3_reduce = self.conv3x3_reduce(x)
		out_3x3 = self.conv3x3(out_3x3_reduce)

		# 5x5 Conv Output
		out_5x5_reduce = self.conv5x5_reduce(x)
		out_5x5 = self.conv5x5(out_5x5_reduce)

		# Max Pool Output
		out_pool = self.pool_proj(self.pool(x))

		return torch.cat((out_1x1, out_3x3, out_5x5, out_pool), dim=1)

class InceptionV3(nn.Module):
	def __init__(self):
		super(InceptionV3, self).__init__()
		self.batch_size = 32
		self.lr_rate = 0.045
		self.layers = nn.ModuleList([
			Conv2D(3, 32, 3, stride=2),
			Conv2D(32, 32, 3),
			Conv2D(32, 64, 3),
			nn.MaxPool2d(3, stride=2),
			Conv2D(64, 80, 1),
			Conv2D(80, 192, 3),
			nn.MaxPool2d(3, stride=2)
		])

		# Inception block a with spatial reduction at the end
		self.layers.extend([
			InceptionBlockA(192, 32),
			InceptionBlockA(256, 64),
			InceptionBlockA(288, 64),
			InceptionBlockB(288)
		])

		# Inception block c with spatial reduction at the end
		self.layers.extend([
			InceptionBlockC(768, 128),
			InceptionBlockC(768, 160),
			InceptionBlockC(768, 160),
			InceptionBlockC(768, 192),
			InceptionBlockD(768)
		])

		# Inception block e
		self.layers.extend([
			InceptionBlockE(1280),
			InceptionBlockE(2048)
		])

		# FC layers
		self.layers.extend([
			nn.AdaptiveAvgPool2d(1),
			nn.Dropout(p=0.5),
			nn.Flatten(),
			nn.Linear(2048, 1000)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x


class InceptionV1(nn.Module):
	def __init__(self):
		self.batch_size = 128
		super(InceptionV1, self).__init__()
		self.layers = nn.ModuleList([
			Conv2D(3, 64, 7, stride=2, pad=3),
			nn.MaxPool2d(3, stride=2, padding=1),
			nn.Conv2d(64, 64, 1),
			nn.Conv2d(64, 192, 3, pad=1),
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
			nn.Flatten(),
			nn.Linear(1024, 1000)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

	def save_checkpoint(self, epoch, optimizer):
		torch.save({
			"epoch": epoch, 
			"model_state_dict": self.state_dict(), 
			"optimizer_state_dict": optimizer.state_dict()
			}, CKPT_PATH)

	def load_checkpoint(self, optimizer):
		checkpoint = torch.load(CKPT_PATH)
		self.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		self.train()

		return checkpoint["epoch"]

	def load_model_weights(self):
		checkpoint = torch.load(CKPT_PATH)
		self.load_state_dict(checkpoint["model_state_dict"])
		self.eval()

if __name__ == "__main__":
	inceptionv1 = InceptionV1()
	print(inceptionv1)