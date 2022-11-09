import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import cv2 as cv
import imutils
from ProcessImage import *
model = load_model('bestmodel.h5')


def predictTumor(image):
    new_image = crop_brain_contour(image)

    image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    image = image / 255.

    image = image.reshape((1, 240, 240, 3))

    res = model.predict(image)

    return res