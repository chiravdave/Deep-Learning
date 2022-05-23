from torch import nn


class Conv2D(nn.Module):
	def __init__(
		self, in_channels, out_channels, kernel, stride=1, pad=0, groups=1, 
		act="relu"):
		super(Conv2D, self).__init__()
		self.layers = nn.ModuleList([
			nn.Conv2d(in_channels, out_channels, kernel, stride=stride, 
				padding=pad, groups=groups, bias=False),
			nn.BatchNorm2d(out_channels)
		])
		self._add_activation(act)

	def _add_activation(self, act):
		if act == "relu":
			self.layers.append(nn.ReLU())
		elif act == "relu6":
			self.layers.append(nn.ReLU6())

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

class DWSConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, pad=1):
		super(DWSConv2D, self).__init__()
		self.depth_wise_conv = Conv2D(
			in_channels, in_channels, 3, stride, pad, in_channels)
		self.point_wise_conv = Conv2D(in_channels, out_channels, 1)

	def forward(self, x):
		return self.point_wise_conv(self.depth_wise_conv(x))

class Bottleneck(nn.Module):
	def __init__(self, in_channels, out_channels, stride, expansion=6):
		super(Bottleneck, self).__init__()
		self.use_skip_connection = stride == 1 and in_channels == out_channels
		if expansion > 1:
			self.expansion = Conv2D(
				in_channels, in_channels * expansion, 1, act="relu6")
		self.dwconv = Conv2D(
			in_channels * expansion, in_channels * expansion, 3, stride, 1, 
			groups=in_channels * expansion, act="relu6")
		self.reduction = Conv2D(
			in_channels * expansion, out_channels, 1, act="none")

	def forward(self, x):
		source = x
		if self.expansion is not None:
			x = self.expansion(x)
		out_reduction = self.reduction(self.dwconv(x))

		return out_reduction + x if self.use_skip_connection else out_reduction

class MobileNetV1(nn.Module):
	def __init__(self, extra_deep=True):
		super(MobileNetV1, self).__init__()
		self.layers = nn.ModuleList([
			Conv2D(3, 32, 3, 2, 1), # O/P: (32, 112, 112)
			DWSConv2D(32, 32), # O/P: (32, 112, 112)
			Conv2D(32, 64, 1), # O/P: (64, 112, 112)
			DWSConv2D(64, 64, 2), # O/P: (64, 56, 56)
			Conv2D(64, 128, 1), # O/P: (128, 56, 56)
			DWSConv2D(128, 128), # O/P: (128, 56, 56)
			Conv2D(128, 128, 1), # O/P: (128, 56, 56)
			DWSConv2D(128, 128, 2), # O/P: (128, 28, 28)
			Conv2D(128, 256, 1), # O/P: (256, 28, 28)
			DWSConv2D(256, 256), # O/P: (256, 28, 28)
			Conv2D(256, 256, 1), # O/P: (256, 28, 28)
			DWSConv2D(256, 256, 2), # O/P: (256, 14, 14)
			Conv2D(256, 512, 1) # O/P: (512, 14, 14)
		])
		if extra_deep:
			self.layers.extend([
				DWSConv2D(512, 512), # O/P: (512, 14, 14)
				Conv2D(512, 512, 1) # O/P: (512, 14, 14)
			] * 5)
		self.layers.extend([
		    DWSConv2D(512, 512, 2), # O/P: (512, 7, 7)
		    Conv2D(512, 1024, 1), # O/P: (1024, 7, 7)
		    DWSConv2D(1024, 1024, 2, 4), # O/P: (1024, 7, 7)
		    Conv2D(1024, 1024, 1), # O/P: (1024, 7, 7)
		    nn.AdaptiveMaxPool2d(1), # O/P: (1024, 1, 1)
		    nn.Flatten(),
		    nn.Linear(1024, 1000)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

class MobileNetV2(nn.Module):
	def __init__(self):
		super(MobileNetV2, self).__init__()
		self.batch_size = 96
		self.lr_rate = 0.045
		self.layers = nn.ModuleList([
			Conv2D(3, 32, 3, 2, 1) # O/P: (32, 112, 112)
		])
		# Bottleneck layers
		self.layers.extend(self._add_bottleneck_layers(32))
		# Final layers
		self.layers.extend([
			Conv2D(320, 1280, 1, act="relu6"), # O/P: (1280, 7, 7)
			nn.AdaptiveMaxPool2d(1), # O/P: (1280, 1, 1)
			nn.Flatten(),
			nn.Linear(1280, 1000)
		])

	def _add_bottleneck_layers(self, in_channels):
		bottleneck_config = [
			# t, c, n, s
			(1, 16, 1, 1), # O/P: (16, 112, 112)
			(6, 24, 2, 2), # O/P: (24, 56, 56)
			(6, 32, 3, 2), # O/P: (32, 28, 28)
			(6, 64, 4, 2), # O/P: (64, 14, 14)
			(6, 96, 3, 1), # O/P: (96, 14, 14)
			(6, 160, 3, 2), # O/P: (160, 7, 7)
			(6, 320, 1, 1) # O/P: (320, 7, 7)
		]
		bottleneck_layers = list()
		for t, c, n, s in bottleneck_config:
			for idx in range(n):
				if idx == 0:
					bottleneck_layers.append(Bottleneck(in_channels, c, s, t))
				else:
					bottleneck_layers.append(Bottleneck(in_channels, c, 1, t))
				in_channels = c

		return bottleneck_layers

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

if __name__ == "__main__":
	mobilenetv1 = MobileNetV1()
	print(mobilenetv1)