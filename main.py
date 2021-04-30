from flask import Flask, request,jsonify
import os
import json

app = Flask(__name__)

@app.route("/predictor",methods=['POST'])
def save_image():
  mriData =request.form["mri"]
  print(mriData)
  return jsonify({'status' : 'Success'})

if __name__ == "__main__":
  app.run(Debug=True)
