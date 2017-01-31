from global_head_file import *
import preprocessing_data
from define_leNet import *


def image_normalization(img): #the function is not yet implemented
    normalized_image = np.copy(img)
    return normalized_image

def train_network(num_iterations, resume = 0):
    y_true = tf.placeholder(tf.float32,shape = [None, classification_num], name = 'y_true')
    y_true_cls = tf.argmax(y_true, dimension = 1)
    NA, raw_output, cost = leNet.get_training_model()
    y_pred = tf.nn.softmax(raw_output)
    y_pred_cls = tf.argmax(y_pred, dimension = 1)
    if resume == 1:
        #optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    else:
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    fileToSave = data_dir + "\model.ckpt"
    error = 1000
    with tf.Session() as sess:
        start = 0
        sess.run(tf.global_variables_initializer())
        if resume == 1:
            saver.restore(sess, fileToSave)
        for i in range(num_iterations):
            total_cost = 0
            start = 0
            max_num_of_file = 221 #this is the data length of folder 0000 which contais the least amount of data of 
            #all catetories
            while start < max_num_of_file - batch_size:
                x_train, y_train = preprocessing_data.getData(image_dir, classification_num, start, batch_size)
                x_train, y_train = shuffle(x_train, y_train)
                feed_dict_train = {global_x: x_train, global_y: y_train}
                nothing, c = sess.run([optimizer, cost], feed_dict = feed_dict_train)
                total_cost += c
                start += batch_size
            print('Epoch', i, 'Completed out of ', num_iterations, 'loss is', total_cost)
            if start >= max_num_of_file - 2* batch_size:
                if total_cost < error:
                    saver.save(sess, fileToSave)
                    error = total_cost
        # fileToSave = os.path.join(dataDir, "StopSignModel.ckpt")
        # print(saver.save(sess,  fileToSave))
        # x_test, y_test_true = getData(pos1, neg, max_num_of_file, 50)
        # feed_dict_train = {x: x_test, y: y_test_true}
        # print(sess.run(accuracy, feed_dict = feed_dict_train))
    sess.close()

def detect_traffic_sign(image): #this image should be a color image of size 32 x 32 
    classification = leNet.get_detection_model()
    y_pred = tf.nn.softmax(classification[0])
    y_cls = tf.argmax(y_pred, dimension = 2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_path = data_dir + "\model.ckpt"
        ckpt = tf.train.get_checkpoint_state(data_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, checkpoint_path)
        classification = sess.run([y_cls], feed_dict={global_x: [image]})
        classification = classification[0][0][0]
        cv2.imshow(traffic_sign_dictionary[classification], image)
        cv2.waitKey(20)
        tf.reset_default_graph()
        sess.close()

def batch_detect_image(dir):
    classification = leNet.get_detection_model()
    y_pred = tf.nn.softmax(classification[0])
    y_cls = tf.argmax(y_pred, dimension = 2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_path = data_dir + "\model.ckpt"
        saver.restore(sess, checkpoint_path)
        #if ckpt and ckpt.model_checkpoint_path:
        for image_name in os.listdir(dir):
            image_path = dir + image_name
            image = cv2.imread(image_path)
            img_resize = cv2.resize(image,(image_size, image_size))
            if color_channels == 1:
                img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                img_resize = np.expand_dims(img_resize, 3)
            
            classification = sess.run([y_cls], feed_dict={global_x: [img_resize]})
            classification = classification[0][0][0]
            cv2.imshow(traffic_sign_dictionary[classification], image)
            print(traffic_sign_dictionary[classification])
            cv2.waitKey(0)
        sess.close()

train_mode = 1
if train_mode == 0:
    dir = r'C:\Users\user\Desktop\test\traffic sign detection data2\00005\\'
    batch_detect_image(dir)
else:
    train_network(50, 1)