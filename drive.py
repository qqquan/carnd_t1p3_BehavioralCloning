import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from util_qDatasetManager import prepImg, image_trim

import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString))) #class 'PIL.JpegImagePlugin.JpegImageFile
    image_array = np.asarray(image)
    image_array = image_array[:,:,::-1]  #convert RGB to BGR to match training with cv2
    if True == Enable_Tiny_Model:
        image_array = image_trim(image_array)
    else:
        image_array = prepImg(image_array)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.

    # if np.absolute(steering_angle) > 0.04:
    #     #curve 
    #     if float(speed) > 10.0:
    #         print('brake: ' )
    #         throttle = -0.5
    #     else:
    #         throttle = 0.1
    # else:
    #     # straight lane
    #     throttle = 0.5
    throttle = 0.5
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

Enable_Tiny_Model = False

if __name__ == '__main__':
    np.random.seed(3721) 
    parser = argparse.ArgumentParser(description='Remote Driving')

    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')

    parser.add_argument("--tiny", default=False, action="store_true" , help="enable tiny model")

    args = parser.parse_args()

    Enable_Tiny_Model = args.tiny

    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)