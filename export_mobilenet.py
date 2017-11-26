
import os

#https://stackoverflow.com/questions/43649591/serving-keras-models-with-tensorflow-serving

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import     build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
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

if FLAGS.model_type == 'mobilenet_1_228_bottleneck':
    model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# The creation of a new model might be optional depending on the goal
config = model.get_config()
weights = model.get_weights()

#Solution from https://github.com/fchollet/keras/issues/7431#issuecomment-334959500
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    new_model = Model.from_config(config)
    new_model.set_weights(weights)


export_path_base = './{}-export'.format(FLAGS.model_type)
export_version = padding = '{0:08d}'.format(FLAGS.version)  # version number (integer)
sess = K.get_session()

# saver = tf.train.Saver(sharded=True)
# model_exporter = exporter.Exporter(saver)
# signature = exporter.classification_signature(input_tensor=model.input,
#                                               scores_tensor=model.output)
#
# model_exporter.init(sess.graph.as_graph_def(),
#                     default_graph_signature=signature)
# model_exporter.export(export_path_base, tf.constant(export_version), sess)

export_path = os.path.join(export_path_base, str(export_version))
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(new_model.input)
tensor_info_y = tf.saved_model.utils.build_tensor_info(new_model.output)

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
    pass


#saver = tf.train.Saver(sharded=True)
#model_exporter = exporter.Exporter(saver)
# signature = exporter.classification_signature(input_tensor=model.input,
#                                               scores_tensor=model.output)
# model_exporter.init(sess.graph.as_graph_def(),
#                     default_graph_signature=signature)
# model_exporter.export(export_path, tf.constant(export_version), sess)


# insights into serving multiple models
# builder = saved_model_builder.SavedModelBuilder(export_path)
#
# signature = predict_signature_def(inputs={'images': new_model.input},
#                                   outputs={'scores': new_model.output})
#
# with K.get_session() as sess:
#     builder.add_meta_graph_and_variables(sess=sess,
#                                          tags=[tag_constants.SERVING],
#                                          signature_def_map={'predict': signature})
#     builder.save()
#

