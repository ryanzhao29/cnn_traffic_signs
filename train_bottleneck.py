from global_head_file import *
import preprocessing_data
from define_leNet import *
saver = tf.train.Saver([leNet.weights_fc1,     leNet.bias_fc1,
                        leNet.weights_fc2,     leNet.bias_fc2,
                        leNet.weights_fc3,     leNet.bias_fc3,
                        leNet.weights_conv1,   leNet.bias_conv1,
                        leNet.weights_conv2,   leNet.bias_conv2,
                        leNet.weights_fc3_FE,  leNet.bias_fc3_FE])

checkpoint_path = data_dir + "\model.ckpt"

def image_normalization(img): #the function is not yet implemented
    normalized_image = np.copy(img)
    return normalized_image

def train_network_BN(num_iterations, resume = 0, learning_rate = 0):
    NA, raw_output, cost, cross_entrophy = leNet.get_training_model(enable_dropout = True, feature_extraction_phase = feature_extraction_status.bottleneck)
    NA1, raw_output_no_dropout, cost1, cross_entrophy1 = leNet.get_training_model(enable_dropout = False,feature_extraction_phase = feature_extraction_status.bottleneck)
    #learning rate
    if learning_rate == 0:
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    #calculate learning accuracy on training set
    y_true_cls = tf.argmax(global_y, dimension = 1)
    y_pred = tf.nn.softmax(raw_output_no_dropout)
    y_pred_cls = tf.argmax(y_pred, dimension = 1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        best_accuracy = 0
        sess.run(tf.global_variables_initializer())
        if resume == 1:
            saver.restore(sess, checkpoint_path)
        for i in range(num_iterations):
            total_cost = 0
            total_entropy = 0
            accuracy_sum = 0
            start = 0
            counter = 0.0
            num_training_sample = 221 #this is the data length of folder 0000 which contais the least amount of data of 
            #all catetories
            while start < num_training_sample:
                x_train, y_train = preprocessing_data.getData(image_dir, classification_num, start, batch_size)
                feed_dict_train = {global_x: x_train, global_y: y_train}
                _, cost_temp, accuracy_temp, entropy_temp = sess.run([optimizer, cost, accuracy, cross_entrophy], feed_dict = feed_dict_train)
                total_cost      += cost_temp
                accuracy_sum    += accuracy_temp
                total_entropy   += entropy_temp
                start           += batch_size
                counter         += 1.0
            average_accuracy = accuracy_sum/counter
            print('Epoch', i, 'of', num_iterations, 'cost is', total_cost, ', entropy is', total_entropy, 'and accuracy is', average_accuracy)
            #save the check point if accuracy improves.
            if average_accuracy > best_accuracy:
                saver.save(sess, checkpoint_path)
                best_accuracy = average_accuracy
            if average_accuracy > 0.96:
                break
    print('the best accuracy over the training set is', best_accuracy)
    sess.close()

def batch_detect_image_BN(dir):
    classification = leNet.get_detection_model(feature_extraction = False)
    y_pred = tf.nn.softmax(classification[0])
    y_top_5 = tf.nn.top_k(y_pred, 2)
    y_cls = tf.argmax(y_pred, dimension = 2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)
        for image_name in os.listdir(dir):
            image_path = dir + image_name
            image = cv2.imread(image_path)
            img_resize = cv2.resize(image,(image_size, image_size))
            if color_channels == 1 :
                img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                img_resize = np.expand_dims(img_resize, 3)
            classification = sess.run([y_cls], feed_dict={global_x: [img_resize]})
            classification = classification[0][0][0]
            # classification = classification[0][1][0][0]
            # if 43 in classification: #if background made to top 5 then treat the image as background image.
            #     classification = 43
            # else:
            #     classification = classification[0]
            cv2.imshow(traffic_sign_dictionary[classification], image)
            print(traffic_sign_dictionary[classification])
            cv2.waitKey(0)
        sess.close()

if __name__ == '__main__':
    train_mode = 0
    if train_mode == 0:
        dir = r'C:\Users\user\Desktop\test\traffic sign detection data2\00002\\'
        #dir = r'C:\Users\user\Desktop\test\speed limit and traffic sign\10\\'
        dir = r'C:\Users\user\Desktop\test\validation stop sign and speed limits\\'
        dir = r'C:\Users\user\Desktop\test\traffic sign detection data_validation\15\\'
        dir = r'D:\doanload\GTSRB\Final_Test\Images\\'
        batch_detect_image_BN(dir)
    else:
        train_network_BN(num_iterations = 300, resume = 1, learning_rate = 2e-4)













#def detect_traffic_sign(image): #this image should be a color image of size 32 x 32 
#    classification = leNet.get_detection_model()
#    y_pred = tf.nn.softmax(classification[0])
#    y_cls = tf.argmax(y_pred, dimension = 2)
#    saver = tf.train.Saver()
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        checkpoint_path = data_dir + "\model.ckpt"
#        ckpt = tf.train.get_checkpoint_state(data_dir)
#        #if ckpt and ckpt.model_checkpoint_path:
#        saver.restore(sess, checkpoint_path)
#        classification = sess.run([y_cls], feed_dict={global_x: [image]})
#        classification = classification[0][0][0]
#        cv2.imshow(traffic_sign_dictionary[classification], image)
#        cv2.waitKey(20)
#        tf.reset_default_graph()
#        sess.close()