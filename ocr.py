import cv2
from PIL import Image, ImageDraw
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from deskew import determine_skew
from mmocr.apis import MMOCRInferencer
from skimage.transform import rotate
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
import math
from typing import Tuple, Union
from skimage.color import rgba2rgb, rgb2gray
import editdistance
from flask import Flask, redirect,render_template,request,jsonify
import requests
from io import BytesIO
import requests
import os
import cv2
import json
import logging
from flask_basicauth import BasicAuth
import time

## logging 
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level to process
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log message format
    handlers=[logging.StreamHandler()]  # Send log messages to the console
)

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'Jiyyo230723'
app.config['BASIC_AUTH_PASSWORD'] = 'Jiyyopass123'
basic_auth = BasicAuth(app)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### model to be installed - > 2.5 gb 
infer = MMOCRInferencer(det='TextSnake')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
model.to(device)



### function to graycsale image adaptively. 
def threshold(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.medianBlur(gray, 1), 300, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7)
    thresh_img = Image.fromarray(thresh)
    return thresh_img


### function to generate text, will be called to generate text by two main functions
def text_from_model(image):
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


### 1. Fucntion to generate text as a list
def generate_text_from_image(img_path):
    text_list = []
    result = infer(img_path, return_vis=True)
    image = Image.open(img_path).convert("RGBA")
    image = threshold(image)

    for itr in range(len(result['predictions'][0]['det_polygons'])):
        polygon_vertices = result['predictions'][0]['det_polygons'][itr]
        polygon_vertices = [(polygon_vertices[i], polygon_vertices[i + 1]) for i in range(0, len(polygon_vertices), 2)]

        min_x = min(vertex[0] for vertex in polygon_vertices)
        max_x = max(vertex[0] for vertex in polygon_vertices)
        min_y = min(vertex[1] for vertex in polygon_vertices)
        max_y = max(vertex[1] for vertex in polygon_vertices)
        bounding_box = (min_x, min_y, max_x, max_y)

        cropped_image = image.crop(bounding_box)

        desired_width = int(max_x - min_x)
        desired_height = int(max_y - min_y)
        cropped_image = cropped_image.resize((desired_width, desired_height), Image.ANTIALIAS)

        text = text_from_model(cropped_image)
        text_list.append(text)
        
    return text_list


### 2. Function to generate text as a dict
def get_word_dict(img_path):
    word_dict = {}
    result = infer(img_path, return_vis=True)
    image = Image.open(img_path).convert("RGBA")

    for itr in range(len(result['predictions'][0]['det_polygons'])):
        polygon_vertices = result['predictions'][0]['det_polygons'][itr]
        polygon_vertices = [(polygon_vertices[i], polygon_vertices[i + 1]) for i in range(0, len(polygon_vertices), 2)]

        min_x = min(vertex[0] for vertex in polygon_vertices)
        max_x = max(vertex[0] for vertex in polygon_vertices)
        min_y = min(vertex[1] for vertex in polygon_vertices)
        max_y = max(vertex[1] for vertex in polygon_vertices)
        bounding_box = (min_x, min_y, max_x, max_y)

        cropped_image = image.crop(bounding_box)

        desired_width = int(max_x - min_x)
        desired_height = int(max_y - min_y)
        cropped_image = cropped_image.resize((desired_width, desired_height), Image.ANTIALIAS)

        text = text_from_model(cropped_image)

        words = text.split()
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    return word_dict


### Examples

img_path = '/kaggle/input/images-7/ocr6.jpg'
output_text = generate_text_from_image(img_path)
output_text

img_path = '/kaggle/input/images-7/ocr6.jpg'
word_dictionary = get_word_dict(img_path)
print("Word Dictionary:", word_dictionary)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 

