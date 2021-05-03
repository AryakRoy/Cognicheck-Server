from flask import Flask, request,jsonify
from predictor import Predictor_Model
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

class Analyzer(Resource):
  def __init__(self):
    parser = reqparse.RequestParser()
    parser.add_argument("mri",type=str,help="MRI URL is required", required=True)
    self.req_parser = parser
    self.pred_model = Predictor_Model()

  def get(self):
    return jsonify({'result' : 'Hello'})
    
  def post(self):
    args = self.req_parser.parse_args()
    mriData = args["mri"]
    result = self.pred_model.analyze(mriData)
    return jsonify({'result' : f'{result}'})

api.add_resource(Analyzer,"/")  

if __name__ == "__main__":
  app.run(debug=True)
