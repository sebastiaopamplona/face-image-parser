import os
import time
import traceback
import multiprocessing
import pickle
from PIL import Image
from mtcnn.mtcnn import MTCNN
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Path to the dataset.
WIKI_DATASET_PATH = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/'
data = loadmat('{}wiki.mat'.format(WIKI_DATASET_PATH))


def reformat_face(face, required_size=(224, 224)):
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    return face_array


def mtcnn_extract_face(pixels):
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]

    return reformat_face(face)


def wiki_extract_face(pixels, coords):
    x1, y1, x2, y2 = coords
    # extract the face
    face = pixels[int(y1):int(y2), int(x1):int(x2)]

    return reformat_face(face)


def to_pickle(obj, filepath):
    pickle_out = open(filepath, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def parse_chunk(process_name, start, end, mtcnn):
    print('[Process {}]:'.format(process_name), end='')
    # 1. Not corrupted; 2. RGB; 3. Contains a face
    valid_face_imgs_filepaths = []
    faces = []
    ages = []
    for i in range(start, end):
        img = data['wiki'][0]['full_path'][0][0][i][0]
        filepath = '{}/{}'.format(WIKI_DATASET_PATH, img)
        try:
            pixels = plt.imread(filepath)
            if (len(pixels.shape) == 3 and pixels.shape[0] != 1 and
                    data['wiki'][0]['face_score'][0][0][i] != (-float('Inf')) and pixels.shape[2] == 3):
                if mtcnn:
                    face = mtcnn_extract_face(pixels)
                else:
                    face = wiki_extract_face(pixels, data['wiki'][0]['face_location'][0][0][i][0])

                age = int(data['wiki'][0]['photo_taken'][0][0][i]) - int(img.split('_')[1].split('-')[0])

                valid_face_imgs_filepaths.append((img,
                                                  data['wiki'][0]['face_location'][0][0][i][0],
                                                  data['wiki'][0]['photo_taken'][0][0][i],
                                                  img.split('_')[1].split('-')[0],
                                                  age))
                faces.append(face)
                ages.append(age)
            else:
                print('[X] Deleting {}'.format(filepath))
                os.remove(filepath) 

        except Exception as e:
            
            pass

    to_pickle(valid_face_imgs_filepaths, 
              'parsed_data/wiki_valid_face_filepaths/valid_face_imgs_filepaths_{}_to_{}.pickle'.format(start, end))
    to_pickle(np.array(faces).reshape(-1, 224, 224, 3),
              'parsed_data/wiki_faces/faces_{}_{}_to_{}.pickle'.format('mtcnn' if mtcnn else 'wiki',
                                                            start,
                                                            end))
    to_pickle(ages, 
              'parsed_data/age/ages_{}_to_{}.pickle'.format(start, end))

    print(' done.')


data_size = len(data['wiki'][0]['full_path'][0][0])
n_chunks = 10
start = int(round(time.time() * 1000))
for i in range(0, n_chunks - 1):
    parse_chunk('p{}'.format(i), int(i * data_size / n_chunks), int((i + 1) * data_size / n_chunks), False)

parse_chunk('p9', int((n_chunks - 1) * data_size / n_chunks), data_size, False)
end = int(round(time.time() * 1000))

print('Time elapsed: {} ms'.format(end - start))

