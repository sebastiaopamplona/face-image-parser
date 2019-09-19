# example of face detection with mtcnn
from keras import Model
from keras.layers import Dense
from matplotlib import pyplot
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import euclidean
import os

img1 = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/00/69300_1950-05-11_2009.jpg'
img2 = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/00/81800_1986-06-13_2011.jpg'
img3 = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/00/37500_1944-01-23_2010.jpg'
img4 = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/00/51800_1942-02-01_2007.jpg'
img5 = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki/05/645405_1963-07-11_2011.jpg'

# def delete_broken():







# extract a single face from a given photograph
# noinspection PyShadowingNames
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
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


# # load the photo and extract the face
# pixels = extract_face(img_filepath)
# # plot the extracted face
# pyplot.imshow(pixels)
# # show the plot
# pyplot.show()

### Example of creating a face embedding

# load the photo and extract the face
# pixels = extract_face(img_filepath)
# # convert one face into samples
# pixels = pixels.astype('float32')
# samples = np.expand_dims(pixels, axis=0)
# # prepare the face for the model, e.g. center pixels
# samples = preprocess_input(samples, version=2)
# # create a vggface model
model2 = VGGFace(model='resnet50')
#
model2.summary()
print(len(model2.layers))

# model.layers.pop()

# new_model = Model(model.inputs, model.layers[-2].output)

# new_model.summary()
# print(len(new_model.layers))

# perform prediction
# yhat = new_model.predict(samples)
# print(yhat.shape)

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)

    print(samples[0].shape)
    print(samples[0])

    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model.summary()
    print(len(model.layers))
    # perform prediction
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = euclidean(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# define filenames
filenames = [img1, img2, img3, img4, img5]
# get embeddings file filenames
embeddings = get_embeddings(filenames)

print(embeddings[0].shape)
