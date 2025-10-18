import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import shutil
import sys

# --- CONFIGURATION ---
DATA_DIR = 'AtoZ_3.1'
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 0
BATCH_SIZE = 16
MODEL_SAVE_PATH = 'cnn_trained_new.keras'
ORIGINAL_MODEL_NAME = 'cnn8grps_rad1_model.h5'

def prepare_data(data_dir):
    global NUM_CLASSES
    classes = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    NUM_CLASSES = len(classes)

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(train_gen, val_gen, model):
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=0,
        mode='max'
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=50,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )

    return history

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    train_generator, validation_generator = prepare_data(DATA_DIR)
    input_shape = IMAGE_SIZE + (3,)
    model = build_cnn_model(input_shape, NUM_CLASSES)

    if train_generator.samples > 0:
        train_model(train_generator, validation_generator, model)

        if os.path.exists(MODEL_SAVE_PATH):
            model.load_weights(MODEL_SAVE_PATH)

        if os.path.exists(MODEL_SAVE_PATH):
            if os.path.exists(ORIGINAL_MODEL_NAME):
                shutil.move(ORIGINAL_MODEL_NAME, 'cnn8grps_rad1_model_OLD.h5')
            shutil.move(MODEL_SAVE_PATH, ORIGINAL_MODEL_NAME)
    else:
        pass
