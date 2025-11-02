# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from model import build_model, predict_emotion
import os

st.set_page_config(page_title='Emotion Detector', layout='centered')

MODEL_WEIGHTS = 'models/emotion_cnn.h5'

@st.cache_resource
def load_resources():
    detector = MTCNN()
    model = build_model()
    if os.path.exists(MODEL_WEIGHTS):
        model.load_weights(MODEL_WEIGHTS)
    return detector, model

detector, model = load_resources()

st.title('Facial Emotion Recognition')
st.write('Upload a photo; the app will detect the largest face and predict the emotion.')

uploaded = st.file_uploader('Choose an image', type=['jpg','jpeg','png'])
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded', use_column_width=True)

    # detect faces
    img_arr = np.array(image)
    faces = detector.detect_faces(img_arr)
    if not faces:
        st.warning('No faces detected')
    else:
        faces = sorted(faces, key=lambda f: f['box'][2]*f['box'][3], reverse=True)
        x,y,w,h = faces[0]['box']
        x, y = max(0,x), max(0,y)
        face = img_arr[y:y+h, x:x+w]
        emotion, conf, probs = predict_emotion(model, Image.fromarray(face))
        st.success(f'Predicted: **{emotion}** ({conf*100:.1f}%)')
        st.write('All probabilities:')
        import pandas as pd
        df = pd.DataFrame({'emotion': ['angry','disgust','fear','happy','sad','surprise','neutral'], 'prob': probs})
        st.dataframe(df)
