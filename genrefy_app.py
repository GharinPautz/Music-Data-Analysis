import os
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Welcome to Genrefy", 200

@app.route("/predict", methods=["GET"])
def predict():
    level = request.args.get("level", "")
    print("level:", level)
    # TODO: actually make the prediction
    result = {"prediction": "True"}
    return jsonify(result), 200

if __name__ == "__main__":
    print("HI")
    port = os.environ.get("PORT", 5000)
    app.run(debug=True)