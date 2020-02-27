import torch
import cv2 as cv2
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from class_names import class_names
import torchvision.transforms as transforms

def resnet50_predict(model, img_path):
    '''
    Use pre-trained resnet-50 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to resnet-50 model's prediction
    '''

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    img = Image.open(img_path)
    img_t = transform(img)
    img = torch.unsqueeze(img_t,0)
    model.cpu()
    
    ## Return the *index* of the predicted class for that image
    output = model(img)
    index = output.data.numpy().argmax() 
    
    return index # predicted class index

def face_detector(face_cascade, img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(model, img_path):
    index = resnet50_predict(model, img_path)
    return (index >= 151) & (index <= 268)   # true/false

def predict_breed(model, img_path):
    # load the image and return the predicted breed
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    img = Image.open(img_path)
    img_t = transform(img)
    img = torch.unsqueeze(img_t,0)
    model.cpu()
    
    out = model(img)
    
    _, index = torch.max(out,1)
    
    return(class_names[index[0]])
