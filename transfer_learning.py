#this file is to use the lenet trained on german traffic sign data for the us traffic data
#need to learn how to perform transfer learning in tensorflow

#this file uses pretrained network(trained somewhere else) and use a new final fc layer to detect new images
#bit resume dictates either training the final layer from scratch or from the last checkpoint
#feature_extraction == True means only train the final fc layer otherwise it is in fine tune mode where  both the fc layer and the pretrained the network are trained

from global_head_file import *
from define_leNet import *
from preprocessing_data import *
from define_neural_network import *
#settings for saver and checkpoint
#--------------------------------------------------------------------------------------
saver_bottleneck = tf.train.Saver([ leNet.weights_fc1,     leNet.bias_fc1,
                                    leNet.weights_fc2,     leNet.bias_fc2,
                                    leNet.weights_fc3,     leNet.bias_fc3,
                                    leNet.weights_conv1,   leNet.bias_conv1,
                                    leNet.weights_conv2,   leNet.bias_conv2 ])

saver_usa_traffic_all = tf.train.Saver([leNet.weights_fc1,       leNet.bias_fc1,
                                        leNet.weights_fc2,       leNet.bias_fc2,
                                        leNet.weights_fc3,       leNet.bias_fc3,
                                        leNet.weights_conv1,     leNet.bias_conv1,
                                        leNet.weights_conv2,     leNet.bias_conv2,
                                        leNet.weights_fc3_FE,    leNet.bias_fc3_FE])
ckpt_file = data_dir + "\model.ckpt"
#--------------------------------------------------------------------------------------
def train_network_FE(num_iterations, resume = False, enable_dropout = True, FE_phase1 = feature_extraction_status.feature_extraction):
    #Dropout should be disabled when running feature extraction.
    fc2, fc3, cost = leNet.get_training_model(enable_dropout = enable_dropout, feature_extraction_phase = FE_phase1)
    y_pred_vector = tf.nn.softmax(fc3)
    y_pred = tf.arg_max(y_pred_vector, dimension = 1)
    #optimizer = tf.train.AdamOptimizer().minimize(cost,var_list = [leNet.weights_fc3_usa, leNet.bias_fc3_usa])
    if FE_phase1 == feature_extraction_status.fine_tune:
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-04).minimize(cost) #small learning rate for fine tune
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-03).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if resume == 1: #load settings of both the pretrained bottlenect feature and the new fc layer
            saver_usa_traffic_all.restore(sess, ckpt_file)
        else:
            saver_bottleneck.restore(sess, ckpt_file) #only load settings of the pretrained bottlenect feature
        for i in range(num_iterations):
            total_cost = 0
            start = 0
            max_num_of_file = 3000
            while start < max_num_of_file:# - batch_size:
                x_train, y_train = getData(image_dir_usa, classification_num_usa, start, batch_size, True, False) #no data augmentation
                feed_dict_train = {global_x: x_train, global_y_FE: y_train}
                nothing, c = sess.run([optimizer, cost], feed_dict = feed_dict_train)
                total_cost += c
                start += batch_size
            print('Epoch', i, 'Completed out of ', num_iterations, 'loss is', total_cost)
        saver_usa_traffic_all.save(sess, ckpt_file)
        sess.close()

def batch_detect_image_FE(dir):
    classification = leNet.get_detection_model(feature_extraction = True)
    y_pred = tf.nn.softmax(classification[0])
    y_cls = tf.argmax(y_pred, dimension = 2)
    top_3 = tf.nn.top_k(y_pred, 3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_usa_traffic_all.restore(sess, ckpt_file)
        for image_name in os.listdir(dir):
            image_path = dir + image_name
            image = cv2.imread(image_path)
            img_resize = cv2.resize(image,(image_size, image_size))
            if color_channels == 1 :
                img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                img_resize = np.expand_dims(img_resize, 2) #tf dose like tensor of 32 x 32, it prefers 32 x 32 x 1, that is what this is for....

            classification, top_3_catg= sess.run([y_cls, top_3], feed_dict={global_x: [img_resize]})
            classification = classification[0][0]
            cv2.imshow(traffic_sign_dictionary_US[classification], image)
            print(traffic_sign_dictionary_US[classification])
            print(top_3_catg)
            cv2.waitKey(0)
        sess.close()
if __name__ == '__main__':
    training_mode = 0
    dir = r'C:\Users\user\Desktop\test\speed limit and traffic sign\5\\'
    dir = r'C:\Users\user\Desktop\test\look alike\\'
    dir = r'C:\Users\user\Desktop\test\validation stop sign and speed limits\\'
    if training_mode:
        train_network_FE(2, resume = True, enable_dropout = True, FE_phase1 = feature_extraction_status.fine_tune)
        #train_network_FE(10, resume=True, enable_dropout=False, FE_phase1=feature_extraction_status.feature_extraction)
    else:
        batch_detect_image_FE(dir)
