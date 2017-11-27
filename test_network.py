import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.saved_model import tag_constants, signature_constants

from keras.applications.mobilenet import decode_predictions, preprocess_input

from export_mobilenet import load_graph, freeze_and_quantize

def read_image_as_nparr_RGB(path, shape = None):
    img_BGR = cv2.imread(path)
    if shape is not None:
        img_BGR = cv2.resize(img_BGR, shape)
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB).astype('float32')




img = np.expand_dims( preprocess_input( read_image_as_nparr_RGB('./elephant.jpeg', shape=(224, 224))), axis=0)

#g = load_graph('./tmp/mobilenet-alpha-1-228-opt.pb')



with tf.Session(graph=tf.Graph()) as sess:
    #op = sess.graph.get_operations()
    # for m in op:
    #     print m.values()
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], './mobilenet-alpha-1-228-export/00000003')

    print '####################\n########################\n################\n'
    op = sess.graph.get_operations()
    for m in op[:100]:
        print m.values()

    for i in range(10):
        s = time.time()
        output = sess.run("reshape_2/Reshape:0", feed_dict={"input_1:0":  img})
        print 'elapsed time', time.time() - s
    print decode_predictions(  output )
