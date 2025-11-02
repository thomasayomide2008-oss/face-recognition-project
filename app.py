# app.py
# Flask app that accepts image uploads and returns predicted emotion for the largest face.
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from model import build_model, predict_emotion, IMG_SIZE

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'png','jpg','jpeg'}
MODEL_WEIGHTS = 'models/emotion_cnn.h5'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load face detector and model
detector = MTCNN()
model = build_model()
if os.path.exists(MODEL_WEIGHTS):
    model.load_weights(MODEL_WEIGHTS)
else:
    print('Warning: model weights not found at', MODEL_WEIGHTS)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error':'no selected file'}), 400
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(path)

        # load with OpenCV
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return jsonify({'error':'invalid image'}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # detect faces
        faces = detector.detect_faces(img_rgb)
        if len(faces) == 0:
            return jsonify({'error':'no face detected'}), 200

        # choose largest face
        faces = sorted(faces, key=lambda f: f['box'][2]*f['box'][3], reverse=True)
        x,y,w,h = faces[0]['box']
        x, y = max(0,x), max(0,y)
        face = img_rgb[y:y+h, x:x+w]
        pil_face = Image.fromarray(face)

        emotion, conf, all_probs = predict_emotion(model, pil_face)

        return jsonify({'emotion':emotion, 'confidence':conf, 'probabilities':all_probs})
    return jsonify({'error':'file type not allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
