import os
import pickle
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to Genrefy</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    sadness = request.args.get("sadness", "")
    feelings = request.args.get("feelings", "")
    danceability = request.args.get("danceability", "")
    loudness = request.args.get("loudness", "")
    accousticness = request.args.get("accousticness", "")
    instumentalness = request.args.get("instrumentalness", "")
    valence = request.args.get("valence", "")
    energy= request.args.get("energy", "")
    
    prediction = predict([sadness, feelings, danceability, loudness, accousticness, instrumentalness, valence, energy])
    if prediction is not None:
        result = {"prediction":prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

def predict(instance):
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True) # TODO: set debug = False
    # app.run(debug = False, host = "0.0.0.0", port = port)