import numpy as np
import os
import pickle
import bisect
import time
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image
from mtcnn.mtcnn import MTCNN

WIKI_DATASET_PATH = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki_crop/'

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

# extract a single face from a given photograph
def mtcnn_extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = plt.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	result = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	try:
		x1, y1, width, height = result[0]['box']
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		plt.imsave(fname='mtcnn_raw_0.png', arr=face)
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = np.asarray(image)
		return face_array
	except Exception as e:
		print('\t MTCNN did not detect a face in {}.'.format(filename))
		return None

def to_pickle(obj, filepath):
    pickle_out = open(filepath, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


valid_filepaths = load_pickles('wiki_valid_face_filepaths', only_first_chunk=False)	
print(len(valid_filepaths))

def mtcnn_extract_chunk(start, end):
	print('start: {}, end: {}'.format(start, end-1))
	data = []
	start_time = int(round(time.time() * 1000))
	for i in range(start, end-1):
		face = mtcnn_extract_face('{}{}'.format(WIKI_DATASET_PATH, valid_filepaths[i][0]))
		age = valid_filepaths[i][4]

		data.append((face, age))

	to_pickle(data, 'chunk_{}_to_{}.pickle'.format(start, end-1))


##############################
#          TESTING           #
##############################
n_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(n_cores)
chunk_size = int(len(valid_filepaths)/n_cores)
tasks = []

for i in range(n_cores):
	tasks.append((i*chunk_size, (i+1)*chunk_size, ))

results = [pool.apply_async(mtcnn_extract_chunk, t) for t in tasks]

start_time = int(round(time.time() * 1000))
for result in results:
    result.get(timeout=10000000)

end_time = int(round(time.time() * 1000))
print('[{} cores] Time elapsed: {} ms'.format(n_cores, end_time - start_time))


