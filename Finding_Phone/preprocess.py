import numpy as np
import cv2

files = []
labels = []
center_values = []

def normalized_testimages(image):
    normalized_Testimage = normalized(image)
    return normalized_Testimage

def normalized_image(filepath, filename):
    labelPath = filepath + filename
    with open(labelPath, 'r') as File:
        infoFile = File.readlines()  # reading lines from files
        for line in infoFile:  # reading line by line
            line2 = line.rstrip('\n')
            words = line2.split(' ')
            files.append(words[0])
	    #Calculating center co-ordinates
            center_values.append(str(float(words[1]) * 256.0))
            center_values.append(str(float(words[2]) * 256.0))
            #Calculating bounding box around the mobile so that later it can be used to mask background
            xmin = int(((float(words[1]) - 0.08) * 256.0) + 0.5)
            xmax = int(((float(words[1]) + 0.08) * 256.0) + 0.5)
            ymin = int(((float(words[2]) - 0.08) * 256.0) + 0.5)
            ymax = int(((float(words[2]) + 0.08) * 256.0) + 0.5)
            labels.append(str(xmin))
            labels.append(str(ymin))
            labels.append(str(xmax))
            labels.append(str(ymax))

    counter = 0
    for file in files:
        imagePath = filepath+file
        img = cv2.imread(imagePath)
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        clone = np.copy(img_resized)
        norm_rgb = normalized(clone)
        clone = np.copy(norm_rgb)
        for x in range(0, 256):
            for y in range(0, 256):
                if (x >= int(labels[counter + 1]) and x <= int(labels[counter + 3]) and y >= int(labels[counter]) and y <= int(labels[counter + 2])):
                    continue
                else:
                    clone[x][y] = 0  #Masking background
        counter = counter + 4
        cv2.imwrite(file, clone)
    return files, center_values

def normalized(img):
    img_clone = img.copy()
    norm = np.zeros((256, 256, 3), np.float32)
    norm_rgb = np.zeros((256, 256, 3), np.uint32)
    b = img_clone[:, :, 0]
    g = img_clone[:, :, 1]
    r = img_clone[:, :, 2]
    sum = np.zeros((256, 256), np.float32)
    for i in range(0, 256):
        for j in range(0, 256):
            sum[i, j] = sum[i, j] + b[i, j] + g[i, j] + r[i, j]
            if sum[i,j] != 0:
                norm[i, j, 0] = b[i, j] / sum[i, j] * 255.0
                norm[i, j, 1] = g[i, j] / sum[i, j] * 255.0
                norm[i, j, 2] = r[i, j] / sum[i, j] * 255.0
	norm_rgb = cv2.convertScaleAbs(norm)
    return norm_rgb
