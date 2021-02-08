import cv2 as cv
import numpy as np
import os
import glob
import face_recognition
import read_write_path as rw
import pickle

# /---- <Path> ----/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
DATA_DIR = os.path.join(BASE_DIR, 'DATA')

temp_path = os.path.join(DATA_DIR, 'train')
train_data = glob.glob(temp_path + '\*.jpg')

temp_path = os.path.join(DATA_DIR, 'test', 'indatabase')
in_database_data = glob.glob(temp_path + '\*.jpg')

temp_path = os.path.join(DATA_DIR, 'test', 'not in database')
not_in_database_data = glob.glob(temp_path + '\*.jpg')
# /---- </Path> ----/

# /---- <Cascades> ----/
haar_cascades_path = r'./tar_hw/DATA/haarcascades/'
face_cascade = cv.CascadeClassifier(haar_cascades_path + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier(haar_cascades_path + 'Cascades/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(haar_cascades_path + 'Cascades/haarcascade_smile.xml')
# /---- </Cascades> ----/

# /---- <Get name ID> ----/
'''
Get name and ID for traning dataset
'''

train_name = [os.path.splitext(os.path.basename(file))[0] for file in
              train_data]  # Extract name of all people in training data set
id_name = list(enumerate(train_name))  # Pair (id, name)

# print(id_name)

# /---- </Get name ID> ----/


# /---- <Image Augmentation> ----/
# (1.) Create folder for each person

# (2.) Image Augmentation

# (3.) Save image


# /---- </Image Augmentation> ----/

# /---- <training> ----/

# name_list = []
# encoding_list = []
# for img_file, name in zip(train_data, train_name):
#     img = cv.imread(img_file)
#     # print(img_file.split(os.path.sep)[-2])
#     # Convert BGR to RGB
#     # OpenCV orders color channels in BGR, but the dlib
#     # actually expects RGB. The face_recognition  module uses dlib ,
#     # so before we proceed, let’s swap color
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     # find/localize the faces of the image resulting in a list of face
#     boxes = face_recognition.face_locations(img, model="cnn")
#     print(boxes)
#     # turn the bounding boxes of the image into a list of 128 numbers i.e. encoding the face into a vector
#     encodings = face_recognition.face_encodings(img, boxes)
#     print(name)
#
#
#     for encoding in encodings:
#         name_list.append(name)
#         encoding_list.append(encoding)
#
#
# data = {'name':name_list,'encoding':encoding_list}
#
# print(rw.save_table_path('encoding_data.pkl'))
#
# with open(rw.save_table_path('encoding_data.pkl'), "wb") as pickle_file:
#     pickle.dump(data, pickle_file)


# /---- <\training> ----/

# /---- <test> ----/
with open(rw.save_table_path('encoding_data.pkl'), "rb") as pickle_file:
    data = pickle.load(pickle_file)

for img_file in not_in_database_data:
    img = cv.imread(img_file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    input_img_name = os.path.splitext(os.path.basename(img_file))[0]  # name of the input file

    boxes = face_recognition.face_locations(img, model="cnn")
    encodings = face_recognition.face_encodings(img, boxes)
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encoding"], encoding)

    face_distances = face_recognition.face_distance(data["encoding"], encoding)
    face_distances = face_distances.tolist()
    print('-' * 100)
    print('input name:', input_img_name)
    print('-' * 100)
    for name, dist in zip(train_name, face_distances):
        print(name + ":\t", dist)
    print('-' * 100)
    print('output name:', train_name[np.argmin(face_distances)])

# /---- <\test> ----/

