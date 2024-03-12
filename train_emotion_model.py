import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the video capture
cap = cv2.VideoCapture(0)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train/Emotion',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'test/Emotion',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

# Define the emotion detection model
model_emotion = Sequential()
model_emotion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)))
model_emotion.add(MaxPooling2D(pool_size=(2, 2)))
model_emotion.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_emotion.add(MaxPooling2D(pool_size=(2, 2)))
model_emotion.add(Flatten())
model_emotion.add(Dense(256, activation='relu'))
model_emotion.add(Dense(7, activation='softmax'))  # Updated to 7 output neurons

# Compile the emotion detection model
model_emotion.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# Train the emotion detection model
model_emotion.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator)

# Save the emotion detection model
model_emotion.save('static/emotion_model.h5')

# Release the video capture
cap.release()