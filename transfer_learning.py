#this file is to use the lenet trained on german traffic sign data for the us traffic data
#need to learn how to perform transfer learning in tensorflow
from global_head_file import *
from define_leNet import *
from preprocessing_data import *
from define_neural_network import *
def feature_extraction():
    fc2, NA, cost = leNet.get_training_model() #here the fc2 is the output of the second fc layer
    weights_fc3_usa = new_weights([leNet.num_fc_layer2, classification_num_us])
    bias_fc3_usa = new_bias(classification_num_us)
    y_raw = tf.matmul(fc2, weights_fc3_usa) + bias_fc3_usa
    y_pred_vector = tf.nn.softmax(y_raw)
    y_pred = tf.arg_max(y_pred_vector, dimension = 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_pred_vector, global_y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cost,var_list = [weights_fc3_usa, bias_fc3_usa])
    saver = tf.train.Saver()
    ckpt_file = data_dir + "\model.ckpt"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_file)


feature_extraction()



