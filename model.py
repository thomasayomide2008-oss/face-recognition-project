# model.py
# Defines a compact CNN for emotion recognition with helper preprocess functions.
import os
import numpy as np
# model.py
# Defines a compact CNN for emotion recognition with helper preprocess functions.
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']
IMG_SIZE = (48,48)

def build_model(input_shape=(48,48,1), n_classes=len(EMOTIONS)):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess_face(img, target_size=IMG_SIZE):
    # img: PIL Image or numpy array (RGB/BGR/greyscale)
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
    else:
        pil = img
    gray = pil.convert('L').resize(target_size)
    arr = img_to_array(gray)
    arr = arr.astype('float32')/255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    arr = np.expand_dims(arr, axis=-1) # channel dim
    return arr


def predict_emotion(model, face_img):
    # face_img: PIL Image or numpy array cropped to face
    x = preprocess_face(face_img)
    preds = model.predict(x)
    idx = int(np.argmax(preds))
    return EMOTIONS[idx], float(preds[0,idx]), preds[0].tolist()


if __name__ == '__main__':
    # quick test: build model and print summary
    m = build_model()
    m.summary()
