import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the video capture
cap = cv2.VideoCapture(0)

# Define the gender labels
gender_labels = ['Male', 'Female']

# Define the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train/Gender',
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'test/Gender',
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical')

# Define the gender detection model
model_gender = Sequential()
model_gender.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96, 96, 3)))
model_gender.add(MaxPooling2D(pool_size=(2, 2)))
model_gender.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_gender.add(MaxPooling2D(pool_size=(2, 2)))
model_gender.add(Flatten())
model_gender.add(Dense(256, activation='relu'))
model_gender.add(Dense(2, activation='softmax'))

# Compile the gender detection model
model_gender.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# Train the gender detection model
model_gender.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator)

# Save the gender detection model
model_gender.save('static/gender_model.h5')

# Release the video capture
cap.release()