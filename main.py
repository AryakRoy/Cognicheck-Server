from flask import Flask, request,jsonify
import os
import json
from predictor import Predictor 

app = Flask(__name__)
Pred = Predictor()

@app.route("/predictor",methods=['POST'])
def save_image():
  mriData =request.form["mri"]
  result = Pred.analyze(mriData)
  return jsonify({'result' : f'{result}'})

if __name__ == "__main__":
  app.run(debug=True)
