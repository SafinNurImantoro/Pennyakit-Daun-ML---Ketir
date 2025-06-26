import numpy as np
import tensorflow as tf
from PIL import Image
import os

def load_model():
    return tf.keras.models.load_model('saved_model/leaf_model.h5')

def predict_image(model, image, class_indices):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    idx = np.argmax(prediction)
    label = list(class_indices.keys())[list(class_indices.values()).index(idx)]
    confidence = prediction[idx]
    return label, confidence

def load_label_map():
    label_map = {}
    if os.path.exists("saved_model/label_map.txt"):
        with open("saved_model/label_map.txt", "r") as f:
            for line in f:
                label, idx = line.strip().split(",")
                label_map[label] = int(idx)
    return label_map
