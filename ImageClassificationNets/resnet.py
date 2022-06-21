from torch import nn

CKPT_PATH = "model.pt"


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, downsample):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=downsample, 
			padding=1, bias=False)
		self.batch_norm1 = nn.BatchNorm2d(out_channels)
		self.act1 = nn.ReLU()
		self.conv2 = nn.Conv2d(
			out_channels, out_channels, 3, stride=1, padding=1, bias=False)
		self.batch_norm2 = nn.BatchNorm2d(out_channels)
		self.act2 = nn.ReLU()

		# Skip Connection Config
		if downsample == 2:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 3, stride=downsample, 
					padding=1, bias=False),
				nn.BatchNorm2d(out_channels)
				)
		else:
			self.downsample = None

		def forward(self, x):
			conv1_out = self.conv1(x)
			batch_norm1_out = self.batch_norm1(conv1_out)
			act1_out = self.act1(batch_norm1_out)
			conv2_out = self.conv2(act1_out)
			batch_norm2_out = self.batch_norm2(conv2_out)

			# Applying skip connection
			if self.downsample:
				x = self.downsample(x)

			return self.act2(batch_norm2_out + x)

class ResNet(nn.Module):
	def __init__(self, depth):
		config = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}
		if depth not in config:
			raise Exception(f"Supported depths are: {list(config.keys())}")
		super(ResNet, self).__init__()
		self.batch_size = 256
		self.lr_rate = 1e-1
		self.layers = nn.ModuleList([
			nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1)
		])

		# Conv2 block
		self.layers.extend(self._create_conv_block(64, 64, config[depth][0], 1))
		# Conv3 block
		self.layers.extend(self._create_conv_block(64, 128, config[depth][1], 2))
		# Conv4 block
		self.layers.extend(self._create_conv_block(128, 256, config[depth][2], 2))
		# Conv1 block
		self.layers.extend(self._create_conv_block(256, 512, config[depth][3], 2))

		# FC layers
		self.layers.extend([
			nn.AvgPool2d(7, stride=1),
			nn.Flatten(),
			nn.Linear(512, 1000)
		])

	def _create_conv_block(self, in_channel, out_channel, repeat, downsample):
		layers = [ResidualBlock(in_channel, out_channel, downsample)]
		for _ in range(1, repeat):
			layers.append(ResidualBlock(out_channel, out_channel, 1))

		return layers

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
	resnet = ResNet(34)
	print(resnet)