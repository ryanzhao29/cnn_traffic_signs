from global_head_file import *
import preprocessing_data
from define_leNet import *

def image_normalization(img): #the function is not yet implemented
    normalized_image = np.copy(img)
    return normalized_image

def train_network(num_iterations, resume = 0):
    y_true = tf.placeholder(tf.float32,shape = [None, classification_num], name = 'y_true')
    y_true_cls = tf.argmax(y_true, dimension = 1)
    raw_output, cost = leNet.get_training_model()
    y_pred = tf.nn.softmax(raw_output)
    y_pred_cls = tf.argmax(y_pred, dimension = 1)

    if resume == 1:
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)
        #optimizer = tf.train.AdamOptimizer().minimize(cost)
    else:
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    fileToSave = os.path.join(data_dir, "TrafficSignModel.ckpt")
    error = 1000
    with tf.Session() as sess:
        start = 0
        sess.run(tf.global_variables_initializer())
        if resume == 1:
            saver.restore(sess, fileToSave)

        for i in range(num_iterations):
            total_cost = 0
            start = 0
            max_num_of_file = 221 #this is the data of folder 0000 which contais the least amount of data of 
            #all catetories
            while start < max_num_of_file - batch_size:
                x_train, y_train = preprocessing_data.getData(image_dir, classification_num, start, 5)
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

train_network(10)