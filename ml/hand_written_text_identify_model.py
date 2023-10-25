# -*- coding: utf-8 -*-
"""hand_written_text_identify_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wSAQ6NSuY6uVhrFjTiVK2TP02OKOABml
"""

pip install scikit-learn

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Notebooks/OCR_course/data/Book1.csv')

df.tail()

dir = '/content/drive/MyDrive/Notebooks/OCR_course/data/dataset'

img_size_w =50
img_size_h =20

img_list = []

for img in os.listdir(dir):
  img_array =cv2.imread(os.path.join(dir,img),cv2.IMREAD_GRAYSCALE)
  new_array =cv2.resize(img_array, (img_size_w,img_size_h))
  img_list.append((img,new_array))

#print(img_list[0])

def get_array(name):
  for image,array in img_list:
    if name == image:
      return array

df['Array'] = df['Image'].apply(get_array)

df.tail()

plt.imshow(img_list[1][1], cmap='gray')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df['Label Class'] = df['Label']

label_encoder = LabelEncoder()
df['Label Class'] = label_encoder.fit_transform(df['Label Class'])

print ('total Classes', label_encoder.classes_)
print ('total length of classes' , len(label_encoder.classes_))
output_Classes = len(label_encoder.classes_)

Label_classes = label_encoder.classes_
print(Label_classes)

import pickle

# Specify the file path with the .pkl extension
output_file_path = 'label_encoder.pkl'

# Saving the object to a file using pickle
with open(output_file_path, 'wb') as f:
    pickle.dump(label_encoder, f)

df.tail()

print(df.columns)

onehot = OneHotEncoder()
labels = onehot.fit_transform(df['Label Class'].values.reshape(-1, 1)).toarray()

print(labels[1])

training_set =df['Array']
train_set = []
for img in training_set:
  img = img.reshape(img_size_w,img_size_h,1)
  train_set.append(img)
train_set = np.array(train_set)

train_set.shape

train_labels = df['Label Class'].values

from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten, Dropout, MaxPooling2D,BatchNormalization,GlobalMaxPooling2D
from tensorflow.keras.models import Model

i = Input(shape=(img_size_w,img_size_h,1))
x = Conv2D(32,(3,3),activation = 'relu', padding ='same') (i)
x= BatchNormalization()(x)
x = Conv2D(32,(3,3),activation = 'relu', padding ='same') (x)
x= BatchNormalization()(x)
x= MaxPooling2D((2,2))(x)
x=Dropout(0.2)(x)


x = Conv2D(64,(3,3),activation = 'relu', padding ='same') (x)
x= BatchNormalization()(x)
x = Conv2D(64,(3,3),activation = 'relu', padding ='same') (x)
x= BatchNormalization()(x)
x= MaxPooling2D((2,2))(x)
x=Dropout(0.2)(x)


x = GlobalMaxPooling2D()(x)

x= Flatten()(x)

x = Dropout(0.2)(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(output_Classes,activation='softmax')(x)

model = Model(i,x)

model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print(len(train_set))
print(len(labels))

train_set_length = len(train_set)
labels_length = len(labels)

print(f"Number of samples in train_set: {train_set_length}")
print(f"Number of samples in labels: {labels_length}")

if train_set_length != labels_length:
    print("Mismatch: The number of samples in train_set and labels is different.")
else:
    print("No mismatch: The number of samples in train_set and labels is the same.")

train_set_length = len(train_set)
labels_length = len(labels)

print(f"Number of samples in train_set: {train_set_length}")
print(f"Number of samples in labels: {labels_length}")

r = model.fit(train_set,labels, epochs=500, batch_size=42,validation_split=0.2)

model.save('handwrite_model.h5')

img = '/content/drive/MyDrive/Notebooks/OCR_course/data/dataset/39.jpg'

img_array = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array,(img_size_w,img_size_h))
array = new_array.reshape(-1,img_size_w,img_size_h,1)
pred =model.predict(array)
y=np.argmax(pred)

print(y)
label_encoder.classes_[y]

import cv2
import numpy as np

# Assuming img_size_w, img_size_h, model, and label_encoder are defined as before
# dir is the path to the directory containing all the images you want to predict
dir = '/content/drive/MyDrive/Notebooks/OCR_course/data/dataset'

predicted_values = []

for img_name in os.listdir(dir):
    # Load and preprocess the image
    img_path = os.path.join(dir, img_name)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size_w, img_size_h))
    array = new_array.reshape(-1, img_size_w, img_size_h, 1)

    # Make prediction using the model
    pred = model.predict(array)

    # Get the predicted class index
    y = np.argmax(pred)

    # Get the corresponding class label from label_encoder.classes_
    predicted_class = label_encoder.classes_[y]

    # Append the predicted class label to the list
    predicted_values.append(predicted_class)

print(predicted_values)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Assuming img_size_w, img_size_h, model, and label_encoder are defined as before
# dir is the path to the directory containing all the images you want to predict
dir = '/content/drive/MyDrive/Notebooks/OCR_course/data/dataset'

predicted_values = []

# Set a common figure size for visualization
plt.figure(figsize=(8, 6))

for img_name in os.listdir(dir):
    # Load and preprocess the image
    img_path = os.path.join(dir, img_name)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size_w, img_size_h))
    array = new_array.reshape(-1, img_size_w, img_size_h, 1)

    # Make prediction using the model
    pred = model.predict(array)

    # Get the predicted class index
    y = np.argmax(pred)

    # Get the corresponding class label from label_encoder.classes_
    predicted_class = label_encoder.classes_[y]

    # Append the predicted class label to the list
    predicted_values.append(predicted_class)

    # Visualize the image with the predicted label
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

print(predicted_values)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Assuming img_size_w, img_size_h, model, and label_encoder are defined as before
# dir is the path to the directory containing all the images you want to predict
dir = '/content/drive/MyDrive/Notebooks/OCR_course/data/dataset'

# Define the size of the grid (3 rows x n_cols)
n_cols = 3

# Initialize an empty list to store the predicted values
predicted_values = []

# Get the number of images in the directory
num_images = len(os.listdir(dir))

# Calculate the number of rows required for the grid
n_rows = (num_images + n_cols - 1) // n_cols

# Create a new figure
plt.figure(figsize=(10, 7))

for idx, img_name in enumerate(os.listdir(dir)):
    # Load and preprocess the image
    img_path = os.path.join(dir, img_name)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size_w, img_size_h))
    array = new_array.reshape(-1, img_size_w, img_size_h, 1)

    # Make prediction using the model
    pred = model.predict(array)

    # Get the predicted class index
    y = np.argmax(pred)

    # Get the corresponding class label from label_encoder.classes_
    predicted_class = label_encoder.classes_[y]

    # Append the predicted class label to the list
    predicted_values.append(predicted_class)

    # Plot the image with the predicted label in a subplot
    plt.subplot(n_rows, n_cols, idx + 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

print(predicted_values)