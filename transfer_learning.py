#this file is to use the lenet trained on german traffic sign data for the us traffic data
#need to learn how to perform transfer learning in tensorflow
from global_head_file import *
from define_leNet import *
from preprocessing_data import *
from define_neural_network import *
def transfer_learning_train(num_iterations, feature_extraction = True):
    fc2, NA, cost = leNet.get_training_model() #here the fc2 is the output of the second fc layer
    if feature_extraction == True: #only train the new fc layer, not the pretrained layer
        fc2 = tf.stop_gradient(fc2)
    leNet.weights_fc3_usa = new_weights([leNet.num_fc_layer2, classification_num_usa])
    leNet.bias_fc3_usa = new_bias(classification_num_usa)
    y_raw = tf.matmul(fc2, leNet.weights_fc3_usa) + leNet.bias_fc3_usa 
    y_pred_vector = tf.nn.softmax(y_raw)
    y_pred = tf.arg_max(y_pred_vector, dimension = 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_pred_vector, global_y_usa)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cost,var_list = [leNet.weights_fc3_usa, leNet.bias_fc3_usa])

    saver_bottleneck = tf.train.Saver([ leNet.weights_fc1,     leNet.bias_fc1,
                                        leNet.weights_fc2,     leNet.bias_fc2,
                                        leNet.weights_fc3,     leNet.bias_fc3,
                                        leNet.weights_conv1,   leNet.bias_conv1 ,
                                        leNet.weights_conv2,   leNet.bias_conv2 ])

    saver_usa_traffic_all = tf.train.Saver([leNet.weights_fc1,       leNet.bias_fc1,
                                            leNet.weights_fc2,       leNet.bias_fc2,
                                            leNet.weights_fc3,       leNet.bias_fc3,
                                            leNet.weights_conv1,     leNet.bias_conv1 ,
                                            leNet.weights_conv2,     leNet.bias_conv2,
                                            leNet.weights_fc3_usa,   leNet.bias_fc3_usa])

    ckpt_file_bottleneck = data_dir + "\model.ckpt"
    ckpt_file_usa = data_dir + "\model_usa.ckpt"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_bottleneck.restore(sess, ckpt_file_bottleneck)
        for i in range(num_iterations):
            total_cost = 0
            start = 0
            max_num_of_file = 1000
            while start < max_num_of_file:# - batch_size:
                x_train, y_train = getData(image_dir_usa, classification_num_usa, start, batch_size, True, False) #no data augmentation
                feed_dict_train = {global_x: x_train, global_y_usa: y_train}
                nothing, c = sess.run([optimizer, cost], feed_dict = feed_dict_train)
                total_cost += c
                start += batch_size
            print('Epoch', i, 'Completed out of ', num_iterations, 'loss is', total_cost)
        saver_usa_traffic_all.save(sess, ckpt_file_usa)
        sess.close()


transfer_learning_train(10)



