import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copy2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "ckplus/CK+48")
test_dirs = os.path.join(BASE_DIR, "ckplus/test")
training_dirs = os.path.join(BASE_DIR, "ckplus/training")
test_dataset = {}
training_dataset = {}


def match_files_to_labels():
    all_files_labels = {}
    label_ids = {}
    current_id = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                all_files_labels[path] = label
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
    return all_files_labels, transform_data_dictionary(all_files_labels)


def transform_data_dictionary(data_dict):
    links, categories = [], []
    for key, value in data_dict.items():
        links.append(key)
        categories.append(value)
    return {"links": links, "categories": categories}


def create_dataframe():
    all_files_labels, transformed_data = match_files_to_labels()
    frame = pd.DataFrame.from_dict(transformed_data)
    return frame


def split_datasets():
    data_frame = create_dataframe()
    y = data_frame.categories
    x_train, x_test, y_train, y_test = train_test_split(
        data_frame, y, test_size=0.2)
    print(">>>>", x_train.shape, x_test.shape, y_train.shape, y_test.shape,
          type(x_train))
    return x_train.to_dict('list'), x_test.to_dict('list')
#
#
# def split_files_into_sets():
#     if not os.path.exists(test_dirs) or not os.path.exists(training_dirs):
#         os.mkdir(test_dirs)
#         os.mkdir(training_dirs)
#
#         for path in test_dataset["links"]:
#             copy2(path, test_dirs)
#         for path in training_dataset["links"]:
#             copy2(path, training_dirs)


