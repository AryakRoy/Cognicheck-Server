import cv2
import imageio
import imutils
import numpy as np
from keras.models import model_from_json
import ssl

class Predictor:
    def crop_brain_contour(self,image, plot=False):
        # Convert the image to grayscale, and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # Find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # Find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        # crop new image out of the original image using the four extreme points (left, right, top, bottom)
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
        json_file = open('best_model.json', 'r')
        model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(model)
        loaded_model.load_weights("best_model.h5")
        print("Loaded model from disk")
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        y_pred = loaded_model.predict(X)
        result = np.where(y_pred>0.7,"Positive","Negative")[0][0]
        print(result)
        return result