FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code

ENTRYPOINT ["python", "genrefy_app.py"]