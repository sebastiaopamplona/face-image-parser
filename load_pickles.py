import os
import bisect
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_pickles(data_type):
    data = []
    parsed_data = []
    files = []
    for filename in os.listdir('parsed_data/{}'.format(data_type)):
        bisect.insort(files, filename)

    for file in files:
        data = pickle.load(open('parsed_data/{}/{}'.format(data_type, file), "rb"))
        for ele in data:
            parsed_data.append(ele)

    return parsed_data


age = load_pickles('age')
wiki_faces = load_pickles('wiki_faces')

# arr = [1, 2, 3]
# arr.pop(0)
# print(arr)


print(len(age))
print(len(wiki_faces))
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

