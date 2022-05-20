from torch import nn


class VGGNet(nn.Module):
	def __init__(self, n_layers=16):
		super(VGGNet, self).__init__()
		n_end_conv_layers = 2 if n_layers == 16 else 3
		self.batch_size = 256
		self.lr_rate = 1e-2
		self.layers = nn.ModuleList([
			nn.Conv2d(3, 64, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, padding="valid"),
			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, padding="valid"),
			nn.Conv2d(128, 256, 3, stride=1, padding=1),
			nn.ReLU()
		])
		self.layers.extend([
			nn.Conv2d(256, 256, 3, stride=1, padding=1), 
			nn.ReLU()
		] * n_end_conv_layers)
		self.layers.extend([
			nn.MaxPool2d(2, stride=2, padding="valid"), 
			nn.Conv2d(256, 512, 3, stride=1, padding=1),
			nn.ReLU()
		])
		self.layers.extend([
			nn.Conv2d(512, 512, 3, stride=1, padding=1), 
			nn.ReLU()
		] * n_end_conv_layers)
		self.layers.append(nn.MaxPool2d(2, stride=2, padding="valid"))
		self.layers.extend([
			nn.Conv2d(512, 512, 3, stride=1, padding=1), 
			nn.ReLU()
		] * (n_end_conv_layers+1))
		self.layers.extend([
			nn.MaxPool2d(2, stride=2, padding="valid"),
			nn.Flatten(),
			nn.Linear(25088, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 1000)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

if __name__ == "__main__":
	vgg16 = VGGNet()
	print(vgg16)