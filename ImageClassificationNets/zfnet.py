from torch import nn


class ZFNet(nn.Module):
	def __init__(self):
		super(ZFNet, self).__init__()
		self.batch_size = 128
		self.lr_rate = 1e-2
		self.layers = nn.ModuleList([
			nn.Conv2d(3, 96, 7, stride=2, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1),
			nn.Conv2d(96, 256, 5, stride=2, padding="valid"),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1),
			nn.Conv2d(256, 384, 3, stride=1, padding="same"),
			nn.ReLU(),
			nn.Conv2d(384, 384, 3, stride=1, padding="same"),
			nn.ReLU(),
			nn.Conv2d(384, 256, 3, stride=1, padding="same"),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding="valid"),
			nn.Flatten(),
			nn.Linear(9216, 4096),
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
	zfnet = ZFNet()
	print(zfnet)