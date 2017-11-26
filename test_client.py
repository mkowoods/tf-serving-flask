import tensorflow as tf
import cv2
import numpy as np
import time
import pickle
from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from keras.applications.mobilenet import preprocess_input, decode_predictions

tf.app.flags.DEFINE_string('server', '35.197.37.75:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', './elephant.jpeg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def read_image_as_nparr_RGB(path, shape = None):
    img_BGR = cv2.imread(path)
    if shape is not None:
        img_BGR = cv2.resize(img_BGR, shape)
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB).astype('float32')


#host, port = '35.197.37.75:9000'.split(':')

def main(_):

    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    #with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.

    img = read_image_as_nparr_RGB(FLAGS.image, shape=(224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    for _ in range(10):
        s = time.time()

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mobilenet-alpha-1-228'
        request.model_spec.signature_name = 'predict'
        request.inputs['images'].CopyFrom(
            tf.make_tensor_proto(img, shape=img.shape)
        )

        result = stub.Predict(request, 10.0)  # 10 secs timeout
        sh = len(decode_predictions( tf.contrib.util.make_ndarray(result.ListFields()[0][1].get('scores')) ))

        #print(result)
        print 'Total Run Time:', time.time() - s, sh


    for _ in range(10):
        s = time.time()

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mobilenet-alpha-1-228-bottleneck'
        request.model_spec.signature_name = 'predict'
        request.inputs['images'].CopyFrom(
            tf.make_tensor_proto(img, shape=img.shape)
        )

        result = stub.Predict(request, 10.0)  # 10 secs timeout
        sh = tf.contrib.util.make_ndarray(result.ListFields()[0][1].get('features')).shape
        #print(result)
        print 'Total Run Time:', time.time() - s, sh


if __name__ == '__main__':

  tf.app.run()
