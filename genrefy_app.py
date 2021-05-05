import os
import pickle
import json
from flask import Flask, jsonify, request
from mysklearn.myclassifiers import MyNaiveBayesClassifier

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to Genrefy</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    dating = request.args.get("dating", "")
    violence = request.args.get("violence", "")
    world_life = request.args.get("world_life", "")
    night_time = request.args.get("night_time", "")
    shake_the_audience = request.args.get("shake_the_audience", "")
    family_gospel = request.args.get("family_gospel", "")
    romantic = request.args.get("romantic", "")
    communication = request.args.get("communication", "")
    obscene = request.args.get("obscene", "")
    music = request.args.get("music", "")
    movement_places = request.args.get("movement_places", "")
    light_visual_perceptions = request.args.get("light_visual_perceptions", "")
    family_spiritual = request.args.get("family_spiritual", "")
    like_girls = request.args.get("like_girls", "")
    sadness = request.args.get("sadness", "")
    feelings = request.args.get("feelings", "")
    danceability = request.args.get("danceability", "")
    loudness = request.args.get("loudness", "")
    accousticness = request.args.get("accousticness", "")
    instumentalness = request.args.get("instrumentalness", "")
    valence = request.args.get("valence", "")
    energy= request.args.get("energy", "")
    age = request.args.get("age", "")


    # get data to fit
    table = mpt.MyPyTable().load_from_file("tcc_ceds_music.csv")

    new_table = myutils.get_even_classifier_instances(table)
    new_table = myutils.categorize_values(new_table)

    # create naive bayes classifier
    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit()

    prediction = naivebayes.predict([dating, violence, world_life, night_time, shake_the_audience, family_gospel, romantic, \
        communication, obscene, music, movement_places, light_visual_perceptions, family_spiritual, like_girls, sadness, \
        feelings, danceability, loudness, accousticness, instrumentalness, valence, energy, age])
    
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