import cv2
import os
from scipy.io import loadmat

data = loadmat('/home/sebastiao/Desktop/DEV/Datasets/wiki/wiki.mat')

for k in range(0, 10):
    img = data['wiki'][0]['full_path'][0][0][k][0]
    filepath = '/home/sebastiao/Desktop/DEV/Datasets/wiki/{}'.format(img)

    pixels = cv2.imread(filepath)
    print('{} has shape {}.'.format(img, pixels.shape))

    if len(pixels.shape) == 3 and pixels.shape[0] != 1 \
            and data['wiki'][0]['face_score'][0][0][k] != (-float('Inf')) \
            and pixels.shape[2] == 3:
        print('{} is corrupted.'.format(img))

# print(data['wiki'][0]['face_score'][0][0][13] == (-float('Inf')))