from flask import Flask, request,jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import cv2
import imageio
import imutils
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
import ssl

app = Flask(__name__)
api = Api(app)
CORS(app)

class Predictor_Model:
    def __init__(self):
        json_file = open('best_model.json', 'r')
        model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(model)
        loaded_model.load_weights("best_model.h5")
        print("Prediction Loaded model from disk")
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = loaded_model

    def crop_brain_contour(self,image, plot=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        return new_image

    def load_data(self,image_url,image_size = (240,240)):
        X = []
        image_width, image_height = image_size
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        image = imageio.imread(image_url)
        image = self.crop_brain_contour(image, plot=False)
        image = cv2.resize(image, dsize=(image_width, image_height) , interpolation=cv2.INTER_CUBIC)
        image = image / 255.
        X.append(image)
        X = np.array(X)
        return X
    
    def analyze(self,image_url):
        X = self.load_data(image_url)
        y_pred = self.model.predict(X)
        result = np.where(y_pred>0.7,"Positive","Negative")[0][0]
        print(f"Prediction {result}")
        return result

class Classification_Model:
    def __init__(self):
        json_file = open('classification.json', 'r')
        model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(model)
        loaded_model.load_weights("classification.h5")
        print("Classification Loaded model from disk")
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        loaded_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
        self.model = loaded_model
        self.IMG_SIZE = 150
    
    def load_data(self,image_url):
        training_data = []
        img_array = imageio.imread(image_url)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(img_array,(self.IMG_SIZE,self.IMG_SIZE)) 
        training_data.append([new_array])
        X = []
        for features in training_data:
            X.append(features)
        X = np.array(X).reshape(-1,self.IMG_SIZE,self.IMG_SIZE)
        X = X/255.0  
        X = X.reshape(-1,150,150,1)
        return X
    
    def analyze(self,image_url):
        X = self.load_data(image_url)
        y_pred = self.model.predict(X)
        if y_pred[0][0] > 0.65:
            result = "Glioma"
        elif y_pred[0][1] > 0.65:
            result = "Meningioma"
        elif y_pred[0][2] > 0.65:
            result = "No Tumor"
        elif y_pred[0][3] > 0.65:
            result = "Pituitary"
        else:
            result = "Invalid"
        print(f"Classification : {result}")
        return result

pred_model = Predictor_Model()
class_model = Classification_Model()

class Analyzer(Resource):
  def __init__(self):
    parser = reqparse.RequestParser()
    parser.add_argument("mri",type=str,help="MRI URL is required", required=True)
    self.req_parser = parser

  def get(self):
    return jsonify({'result' : 'Hello'})
    
  def post(self):
    args = self.req_parser.parse_args()
    mriData = args["mri"]
    pred_result = pred_model.analyze(mriData)
    if pred_result == "Positive":
        class_result = class_model.analyze(mriData)
    else:
        class_result = "None"
    
    return jsonify({'pred_result' : f'{pred_result}', 'class_result' : f'{class_result}'})

api.add_resource(Analyzer,"/")  

if __name__ == "__main__":
  app.run(debug=True)
