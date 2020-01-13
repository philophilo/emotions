import cv2
import os
import numpy as np
import pickle
from PIL import Image
from create_training_test_datasets import split_datasets


# stable releative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "ckplus/training")
training_dataset, test_dataset = split_datasets()
training_links = training_dataset['links']
training_categories = training_dataset['categories']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

# for root, dirs, files in os.walk(image_dir):
#     for file in files:
#         if file.endswith("png"):
#             path = os.path.join(root, file)
#             label = os.path.basename(root).replace(" ", "-").lower()
#             # print(path, label)

for index, link in enumerate(training_links[:1]):
    image = cv2.imread(link)
	# convert to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = (256, 256)
    final_image = cv2.resize(image, size)
    # kernel = np.ones((5,5), np.float32)/25
    gaussian_blur = cv2.GaussianBlur(final_image, (3,3), .25)
    path = os.path.join("./", "new")
    denoised = cv2.fastNlMeansDenoising(final_image, None, 30.0, 7, 21)
    print(image.shape)
    cv2.imshow('denoised', denoised)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     image_array = np.array(final_image, "uint8")
#     print("---", image_array)
#            if label not in label_ids:
#                label_ids[label] = current_id
#                current_id += 1
#            id_ = label_ids[label]
#            print("---->", label_ids)
#            pil_image = Image.open(path).convert("L")
#            size = (256, 256)
#            final_image = pil_image.resize(size, Image.ANTIALIAS)
#            image_array = np.array(final_image, "uint8")
#            print("---", image_array)
#            faces = face_cascade.detectMultiScale(
#                image_array, scaleFactor=1.5, minNeighbors=5)
#
#            for (x, y, w, h) in faces:
#                roi = image_array[y:y+h, x:x+w]
#                x_train.append(roi)
#                y_labels.append(id_)
#
#with open("labels.pickle", "wb") as f:
#    pickle.dump(label_ids, f)
#
#recogniser.train(x_train, np.array(y_labels))
#recogniser.save("trainer.yml")
## print("###", y_labels)
## print("@@@", x_train)
