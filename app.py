import os, io
import torch
import numpy as np
from PIL import Image
from cv2 import cv2
from model import Net
import torchvision.models as models
from utils import dog_detector, face_detector, predict_breed

from flask import Flask,request,jsonify,render_template

# load model
model = Net()
model.load_state_dict(torch.load('models/model_scratch.pt', map_location='cpu'))

# define VGG16 model
resnet50 = models.resnet50(pretrained=True)

# load face cascade
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def render_page():
    return render_template('dog-web-app.html')

@app.route('/uploadajax',methods=['POST'])
def upload_file():
    
    # retrieve the image uploaded and make sure it is an image file
    file = request.files['file']
    image_extensions=['jpg', 'jpeg', 'png']
    if file.filename.split('.')[1] not in image_extensions:
        return jsonify('Please upload an appropriate image file')

    # Load variables needed to detect human face/dog pil image for dog detection and breed prediction 
    image_bytes = file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))

    if (dog_detector(resnet50, pil_image)):
        dog_breed = predict_breed(model, pil_image)
        return jsonify ('This a dog picture of breed:{}'.format(dog_breed))

    elif (face_detector(face_cascade, pil_image)):
        dog_breed = predict_breed(model, pil_image)
        return jsonify('Hello Human You resemble dog breed of {}'.format(dog_breed))
        
    else:
        return jsonify('This pic doesn\'t have a human face or a dog')


if __name__ == '__main__':
    app.run(debug=True)
