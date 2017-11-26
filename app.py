from flask import Flask
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
import time
import sys

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
app = Flask(__name__)


TF_SERV_HOST = 'localhost'
TF_SERV_PORT = '9000'
TF_SERV_CHANNEL = implementations.insecure_channel(TF_SERV_HOST, int(TF_SERV_PORT))
TF_SERV_STUB = prediction_service_pb2.beta_create_PredictionService_stub(TF_SERV_CHANNEL)

def predict_classes(img):
    s = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mobilenet-alpha-1-228'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(
        make_tensor_proto(img, shape=img.shape)
    )
    print 'size of request', sys.getsizeof( str(request) )
    result = TF_SERV_STUB.Predict(request, 10.0)  # 10 secs timeout
    result = make_ndarray(result.ListFields()[0][1].get('scores'))
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
    print 'size of request', sys.getsizeof( str(request) )
    result = TF_SERV_STUB.Predict(request, 10.0)  # 10 secs timeout
    result = make_ndarray(result.ListFields()[0][1].get('features'))
    print 'elapsed time', time.time() - s
    return result


def read_image_as_nparr_RGB(path, shape = None):
    img_BGR = imread(path)
    if shape is not None:
        img_BGR = resize(img_BGR, shape)
    return cvtColor(img_BGR, COLOR_BGR2RGB).astype('float32')

def _preprocess_image(x):
    assert len(x.shape) == 3, 'only support 3 dimensional arrays'
    # taken from https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
    # 'RGB'->'BGR'
    x = x[::-1, ...]
    # Zero-center by mean pixel
    x[0, :, :] -= 103.939
    x[1, :, :] -= 116.779
    x[2, :, :] -= 123.68

    x = np.expand_dims(x, axis=0)
    return x

@app.route('/api/test/classes')
def test_classes():
    img = read_image_as_nparr_RGB('./elephant.jpeg')
    img = _preprocess_image(img)
    img = predict_classes(img)
    return str(img.shape)

@app.route('/api/test/features')
def test_features():
    img = read_image_as_nparr_RGB('./elephant.jpeg')
    img = _preprocess_image(img)
    feat = get_features(img)
    return str(feat.shape)


if __name__ == "__main__":
    #app.run(debug= True)
    app.run(host = '0.0.0.0', port = 5000)