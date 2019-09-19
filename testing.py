import datetime

from PIL import Image
from mtcnn.mtcnn import MTCNN
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pickle


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)

    # print('>>>>>>>>>>> {}'.format(filename))

    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


data = loadmat('/home/sebastiao/Desktop/DEV/Datasets/wiki/wiki.mat')

# print(len(data['wiki'][0]['face_score'][0][0]))
print(data['wiki'][0]['face_score'][0][0][13] == (-float('Inf')))

# face_location = data['wiki'][0]['face_location'][0][0][0][0]

# for i in range(0,4):
#     print(int(face_location[i]))

# dob = data['wiki'][0]['dob'][0][0][0]
# print(dob)
#
# # split
#
# img = str(data['wiki'][0]['full_path'][0][0][1][0])
# print(img)
# print(int(img.split('_')[1].split('-')[0]))
#
# print(data['wiki'][0]['photo_taken'][0][0][1])


# print(face_location)
# img = data['wiki'][0]['full_path'][0][0][0][0]
# filepath = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/{}'.format(img)
# pixels = plt.imread(filepath)
# cropped = pixels[int(face_location[0]):int(face_location[0]) + int(face_location[2]),
#           int(face_location[1]):int(face_location[1]) + int(face_location[3])]
#
# print('{} has shape {}'.format(img, pixels.shape))
# plt.title(img)
# plt.imshow(pixels)
# plt.show()
#
# plt.title('(cropped){}'.format(img))
# plt.imshow(cropped)
# plt.show()
#
# face = extract_face(filepath)
# plt.imshow(face)
# plt.show()


data_size = len(data['wiki'][0]['full_path'][0][0])
age_data = []
# for k in range(0, data_size):
for k in range(90, 100):
    print('[{}/{}]'.format(k, data_size))
    img = data['wiki'][0]['full_path'][0][0][k][0]
    filepath = '/home/sebastiao/Desktop/DEV/Datasets/wiki/{}'.format(img)
    pixels = plt.imread(filepath)
    print('\t{} has shape {}'.format(img, pixels.shape))
    # print('{}'.format(img))
    if len(pixels.shape) == 3 and pixels.shape[0] != 1 \
            and data['wiki'][0]['face_score'][0][0][k] != (-float('Inf')) \
            and pixels.shape[2] == 3:
        # print('rgb and not broken')
        try:
            age = int(data['wiki'][0]['photo_taken'][0][0][k]) - int(img.split('_')[1].split('-')[0])
            age_data.append([extract_face(filepath), age])
        except Exception as e:
            pass
    else:
        pass
        # print('black & white or broken'.format(pixels.shape[0]))

X = []
y_age = []

for img, age in age_data:
    X.append(img)
    y_age.append(age)

X = np.array(X).reshape(-1, 224, 224, 3)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_age.pickle", "wb")
pickle.dump(y_age, pickle_out)
pickle_out.close()
