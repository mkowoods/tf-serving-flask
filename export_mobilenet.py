
import os

#https://stackoverflow.com/questions/43649591/serving-keras-models-with-tensorflow-serving

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf



import keras
from keras.utils.generic_utils import CustomObjectScope
import keras.backend as K
from keras.applications import MobileNet
from keras.models import Model

tf.app.flags.DEFINE_integer('version', '1', 'Model Version')
tf.app.flags.DEFINE_string('model_type', '', 'which model do you want to train')

FLAGS = tf.app.flags.FLAGS


# very important to do this as a first thing
K.set_learning_phase(0)



#if FLAGS.model_type == 'mobilenet_1_228_bottleneck':
#model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
model = MobileNet(weights='imagenet')

# The creation of a new model might be optional depending on the goal
config = model.get_config()
weights = model.get_weights()

#Solution from https://github.com/fchollet/keras/issues/7431#issuecomment-334959500
# with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#     model2 = Model.from_config(config)
#     model2.set_weights(weights)


sess = K.get_session()

def freeze_models(sess, out_name, fpath = 'mobilenet-alpha-1-228.pb'):
    #https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow
    #https://gist.github.com/tokestermw/795cc1fd6d0c9069b20204cbd133e36b


    frozen_graph_def = convert_variables_to_constants(sess, sess.graph_def, [out_name])

    with tf.gfile.GFile("./tmp/" + fpath, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    return frozen_graph_def


#
# def quantize_models(sess, model):
#     """
#     transforms taken from this https://www.tensorflow.org/performance/quantization
#
#     :param sess:
#     :param model:
#     :return:
#     """
#     inputs = [model.input.name.split(':')[0]]
#     outputs = [model.output.name.split(':')[0]]
#     transforms = 'add_default_attributes strip_unused_nodes fold_constants(ignore_errors=true) fold_batch_norms \
#      fold_old_batch_norms quantize_weights quantize_nodes strip_unused_nodes sort_by_execution_order'.split()
#     results = TransformGraph(sess.graph_def, inputs, outputs, transforms)
#
#     return results

def freeze_and_quantize(sess, model, fpath):
    out_name=  model.output.name.split(':')[0]
    frozen_graph_def = convert_variables_to_constants(sess, sess.graph_def, [out_name])

    frozen_graph_def = freeze_models(sess, out_name, fpath + '.pb')

    inputs = [model.input.name.split(':')[0]]
    outputs = [out_name]
    # quantize_weights quantize_node don't quantize mobilenet https://stackoverflow.com/questions/44832492/tensorflow-ssd-mobilenet-model-accuracy-drop-after-quantization-using-transform
    transforms = 'add_default_attributes strip_unused_nodes fold_constants(ignore_errors=true) fold_batch_norms \
     fold_old_batch_norms strip_unused_nodes sort_by_execution_order'.split()
    results = TransformGraph(frozen_graph_def, inputs, outputs, transforms)


    with tf.gfile.GFile('./tmp/'+ fpath+'-opt.pb', 'wb') as f:
        f.write(results.SerializeToString())
    return results


def load_graph(frozen_graph_filename = None):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            name="")
        return graph

def save_model_from_graph(graph, model, export_path = './tmp/test'):

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(model.output)

    with tf.Session(graph=graph) as sess:
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'features': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature
            },
            legacy_init_op=legacy_init_op)
        builder.save()

    print 'Model Save to ', export_path


def ppppp():
    export_path_base = './{}-export'.format(FLAGS.model_type)
    export_version = '{0:08d}'.format(FLAGS.version)  # version number (integer)

    export_path = os.path.join(export_path_base, str(export_version))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(model.output)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'features': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={
               'predict':
                   prediction_signature
          },
        legacy_init_op=legacy_init_op)
    builder.save()

if __name__ == "__main__":
    import shutil
    dir = './mobilenet-alpha-1-228-export/00000001'
    if os.path.isdir(dir): shutil.rmtree(dir)
    freeze_and_quantize(sess, model, 'mobilenet-alpha-1-228')
    g = load_graph('./tmp/mobilenet-alpha-1-228-opt.pb')
    #g = load_graph('./tmp/mobilenet-alpha-1-228-bottleneck-quant.pb')
    save_model_from_graph(g, model, export_path = dir)
