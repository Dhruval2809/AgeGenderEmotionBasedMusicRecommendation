import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the video capture
cap = cv2.VideoCapture(0)

# Define the age labels
age_labels = ['18-20', '21-30', '31-40', '41-50', '51-60']

# Define the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train/Age',
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'test/Age',
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical')

# Define the age detection model
model_age = Sequential()
model_age.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96, 96, 3)))
model_age.add(MaxPooling2D(pool_size=(2, 2)))
model_age.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_age.add(MaxPooling2D(pool_size=(2, 2)))
model_age.add(Flatten())
model_age.add(Dense(256, activation='relu'))
model_age.add(Dense(5, activation='softmax'))

# Compile the age detection model
model_age.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# Train the age detection model
model_age.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator)

# Save the age detection model
model_age.save('static/age_model.h5')

# Release the video capture
cap.release()