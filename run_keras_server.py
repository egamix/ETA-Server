# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

import tensorflow as tf
from tensorflow.python.keras.backend import tanh, conv1d, expand_dims, squeeze, clip
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Flatten, Conv1D, Conv2D, Lambda, concatenate, Reshape, TimeDistributed, MaxPooling1D, Permute, BatchNormalization, ELU
from tensorflow.python.keras.layers.recurrent import LSTM,RNN
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras import losses
# from tensorflow.python.keras.activations import elu
import tensorflow.python.keras.backend as K

from tensorflow import get_default_graph
from tensorflow.keras import models
from joblib import dump, load
from feature_builder import derive_features
import numpy as np
import pandas as pd
import flask
import io
import json

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model, graph, sess
    global km, sc, scy
    tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
    sess = tf.Session()
    graph = get_default_graph()
    K.set_session(sess)
    model = models.load_model('./Model_160920/cnnrnn_160920_SF')
    print('Done')
    km = load('./Model_160920/km_160920_SF.joblib')
    sc = load('./Model_160920/sc_160920_SF.joblib')
    scy = load('./Model_160920/scy_160920_SF.joblib')

def prepare_data(df):
    df2 = derive_features(df, km, None)
    return df2.drop(['time', 'time_elapsed_seconds', 'time_to_dest'],axis = 1 )
    # return df2.drop(['Unnamed: 0', 'time', 'ID', 'time_elapsed_seconds', 'time_to_dest', 'next_lat', 'next_lon'] ,axis =1)
    
    
@app.route("/test")
def test():
    return "it works!"

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        request_json = flask.request.json
        
        # data = json.load(request_json)
        df = pd.read_json(json.dumps(request_json['Data']), convert_dates = ['time'])
        df = df[['time', 'latitude', 'longitude']]
        df['dropoff_latitude'] = request_json['Dropoff'][0]
        df['dropoff_longitude'] = request_json['Dropoff'][1]

        # df = pd.read_json(request_json, convert_dates = ['time'])
        df2 = prepare_data(df)
        x_test = sc.transform(df2.values)
        
        with graph.as_default():
            K.set_session(sess)
            y_pred = model.predict([np.expand_dims(x_test[:,0:2], axis=0), np.expand_dims(x_test[:,2:], axis =0)])
        y_pred_rescaled = scy.inverse_transform(y_pred[0])
        
        data["predictions"] = y_pred_rescaled.flatten().tolist()

        # indicate that the request was a success
        data["success"] = True
    
    # print(data)
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    load_model()
    app.run('0.0.0.0', '5000')