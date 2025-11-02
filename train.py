# train.py
# Example training loop using the FER2013 CSV or an image directory.
# This script is a template — you need to provide training data (FER2013 or organized folders).

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import build_model, EMOTIONS
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# assume you have data/ train/val/ subfolders with emotion-named folders
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
BATCH = 64
EPOCHS = 30
IMG_SIZE = (48,48)

if __name__ == '__main__':
    model = build_model(input_shape=(48,48,1), n_classes=len(EMOTIONS))

    datagen = ImageDataGenerator(rescale=1./255,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    train_gen = datagen.flow_from_directory(TRAIN_DIR,
                                            target_size=IMG_SIZE,
                                            color_mode='grayscale',
                                            batch_size=BATCH,
                                            class_mode='categorical')

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(VAL_DIR,
                                            target_size=IMG_SIZE,
                                            color_mode='grayscale',
                                            batch_size=BATCH,
                                            class_mode='categorical')

    callbacks = [
        ModelCheckpoint('models/emotion_cnn.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)


# Note: To train, download FER2013 or prepare folders labelled by emotion; training GPU recommended.