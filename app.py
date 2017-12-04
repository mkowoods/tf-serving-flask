from flask import Flask, jsonify, request
#import cv2
from PIL import Image
import json
import numpy as np
from tensorflow.python.framework.tensor_util import make_tensor_proto
import time

from PIL import Image
import io

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2 #this library uses a lot of memory

#from keras.applications.mobilenet import preprocess_input, decode_predictions

app = Flask(__name__)

IMAGENET_CLASS_INDEX = json.load(open('imagenet_class_index.json', 'rb'))

TF_SERV_HOST = '35.197.84.177'
TF_SERV_PORT = '9000'
TF_SERV_CHANNEL = implementations.insecure_channel(TF_SERV_HOST, int(TF_SERV_PORT))
TF_SERV_STUB = prediction_service_pb2.beta_create_PredictionService_stub(TF_SERV_CHANNEL)


def predict_classes(img):
    #lazy loading to help with memory management on unused workers

    s = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mobilenet-classify'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(
        make_tensor_proto(img, shape=img.shape)
    )
    #print 'size of request', sys.getsizeof( str(request) )
    result = TF_SERV_STUB.Predict(request, 20.0)  # 10 secs timeout
    print 'time to predict', time.time() - s
    result = np.array([result.outputs['features'].float_val])
    print 'elapsed time', time.time() - s
    return result

def get_features(img):
    s = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mobilenet-alpha-1-228-bottleneck'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(
        make_tensor_proto(img, shape=img.shape)
    )
    #print 'size of request', sys.getsizeof( str(request) )
    result = TF_SERV_STUB.Predict(request, 20.0)  # 10 secs timeout

    features = np.array([result.outputs['features'].float_val])
    print 'elapsed time', time.time() - s
    return features


def _mobilenet_preprocess_image(pil_img, shape = (224, 224)):
    img = pil_img
    if img.size != shape:
        img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img /= 127.5
    img -= 1.
    img = np.expand_dims(img, axis=0)
    return img

def _mobilenet_decode_predictions(class_vector, top_n = 5):
    class_vector = class_vector[0]
    top_n_idx = class_vector.argsort()[-top_n:][::-1]
    return [IMAGENET_CLASS_INDEX[str(idx)] + [class_vector[idx]] for idx in top_n_idx]

TEST_IMG = _mobilenet_preprocess_image(Image.open('elephant.jpeg'))

@app.route('/')
def index():
    return "pong"

@app.route('/api/test/classes')
def test_classes():
    class_vector = predict_classes(TEST_IMG)
    return jsonify({'data': _mobilenet_decode_predictions(class_vector, top_n=25)})

@app.route('/api/test/features')
def test_features():
    feat = get_features(TEST_IMG)
    return jsonify({'data': feat.tolist()})

@app.route('/api/classes', methods = ['POST'])
def api_classes():
    img = request.data
    img = Image.open( io.BytesIO(img))
    img = _mobilenet_preprocess_image(img)
    pred = predict_classes(img)
    return jsonify({'data': _mobilenet_decode_predictions(pred, top_n = 25)})

@app.route('/api/features', methods = ['POST'])
def api_features():
    img = request.data
    img = Image.open( io.BytesIO(img))
    img = _mobilenet_preprocess_image(img)
    pred = get_features(img)
    return jsonify({'data': pred.tolist()})

if __name__ == "__main__":
    app.run(debug= True)
    #app.run(host = '0.0.0.0', port = 5000)
