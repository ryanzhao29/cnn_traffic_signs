#this file is to use the lenet trained on german traffic sign data for the us traffic data
#need to learn how to perform transfer learning in tensorflow
from global_head_file import *
from define_leNet import *
from preprocessing_data import *
from define_neural_network import *
def feature_extraction():
    fc2, NA, cost = leNet.get_detection_model()
    raw_output = new_fc_layer(fc2,leNet.num_fc_layer2, classification_num_us, use_relu = False)
    y_pred_vector = tf.nn.softmax(raw_output)
    y_pred = tf.arg_max(y_pred_vector, y_pred_vector)
    



