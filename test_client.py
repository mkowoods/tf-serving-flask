from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2



tf.app.flags.DEFINE_string('server', '35.197.37.75:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  print host, port
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    data = f.read()
    #print data
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(
        tf.make_tensor_proto(data)
    )
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
  tf.app.run()