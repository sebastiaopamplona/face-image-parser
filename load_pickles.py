import os
import bisect
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
import time

def load_pickles(data_type, only_first_chunk):
    data = []
    parsed_data = []
    files = []
    for filename in os.listdir('parsed_data/{}'.format(data_type)):
        bisect.insort(files, filename)

    for file in files:
        data = pickle.load(open('parsed_data/{}/{}'.format(data_type, file), "rb"))
        for ele in data:
            parsed_data.append(ele)
        if only_first_chunk:
            break

    return parsed_data

cropped_path = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki_crop/'

def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
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

#age = load_pickles('age')
#wiki_faces = load_pickles('wiki_faces', only_first_chunk=True)
valid_filepaths = load_pickles('wiki_valid_face_filepaths', only_first_chunk=False)

for i in range(8):
    data = pickle.load(open('chunk_{}_to_{}.pickle'.format(i*5, (i+1)*5),'rb'))
    for k in range(5):
        plt.imsave(fname='{}_{}.png'.format(i*5+k, data[k][1]), arr=data[k][0])

'''
for i in range(5):
    start = int(round(time.time() * 1000))
    mtcnn_face = extract_face('{}{}'.format(cropped_path, valid_filepaths[i][0]))
    end = int(round(time.time() * 1000))
    print('MTCNN extracted in  {} ms'.format(end - start))
    age = valid_filepaths[i][4]

    print(age)


wiki0, wiki1, wiki2 = wiki_faces[87], wiki_faces[88], wiki_faces[89]
img0, img1, img2 = valid_filepaths[87], valid_filepaths[88], valid_filepaths[89]
print(len(valid_filepaths))
print(img0[0])
print(img1[0])
print(img2[0])


plt.imsave(fname='extract_wiki_0.png', arr=wiki0)
plt.imsave(fname='extract_wiki_1.png', arr=wiki1)
plt.imsave(fname='extract_wiki_2.png', arr=wiki2)



plt.imsave(fname='mtcnn_0.png', arr=extract_face('{}{}'.format(cropped_path, img0[0])))
plt.imsave(fname='mtcnn_1.png', arr=extract_face('{}{}'.format(cropped_path, img1[0])))
plt.imsave(fname='mtcnn_2.png', arr=extract_face('{}{}'.format(cropped_path, img2[0])))
'''
#detector = MTCNN()



# arr = [1, 2, 3]
# arr.pop(0)
# print(arr)


#print(len(age))
#print(len(wiki_faces))
# CHUNK_SIZE = 1000
#
# mtcnn_faces = pickle.load(open("parsed_data/faces_mtcnn_0_to_{}.pickle".format(CHUNK_SIZE), "rb"))
# wiki_faces = pickle.load(open("parsed_data/faces_wiki_0_to_{}.pickle".format(CHUNK_SIZE), "rb"))
# ages = pickle.load(open("parsed_data/ages_0_to_{}.pickle".format(CHUNK_SIZE), "rb"))
#
# for i in range(0, len(mtcnn_faces)):
#     # print(ages_0_to_10[i])
#     plt.title('[MTCNN] Age: {}'.format(ages[i]))
#     plt.imshow(mtcnn_faces[i])
#     plt.show()
#     plt.title('[WIKI] Age: {}'.format(ages[i]))
#     plt.imshow(wiki_faces[i])
#     plt.show()

