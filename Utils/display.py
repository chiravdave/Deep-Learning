import matplotlib.pyplot as plt

from numpy import ndarray

def show_images(images: ndarray, cols: int=3) -> None:
	n_images, idx = images.shape[0], 0
	rows = n_images//cols if n_images%cols == 0 else (n_images//cols)+1
    for row in range(1, rows+1):
    	for _ in range(cols):
    		plt.subplot(row, cols, idx+1)
    		plt.imshow(images[idx])
    		plt.axis("off")
    		idx += 1

    plt.show()