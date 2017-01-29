import tensorflow as tf
import numpy as np
import os
import cv2
import random
import time
image_size = 48

class build_NN(self):
    def new_weights(shape):
        return tf.Variable(tf.random_normal(shape, 0, 0.05))

    def new_bias(length):
        return tf.Variable(tf.random_normal( shape = [length]))

    def new_convolution_layer(input, num_input, filter_size1, filter_size2, num_output, use_pooling = True, padding = 'SAME'):
        #conv 层的的权重的纬度[卷积滤波器宽度, 卷积滤波器宽度, 本层输入数量, 本层输出数量]
        shape = [filter_size1, filter_size2, num_input, num_output] #[filter_size1, filter_size2, num_input,num_output]
        weights = new_weights(shape = shape)
        bias = new_bias(length = num_output)
        layer = tf.nn.conv2d(input = input, filter = weights, strides = [1,1,1,1],padding = padding) + bias
        if use_pooling:
            layer = tf.nn.max_pool(value = layer, ksize = [1,2,2,1],strides = [1,2,2,1],padding = padding)
        layer = tf.nn.relu(layer)
        return layer, weights

    def flatten_layer(layer):
        layer_shape = layer.get_shape()
       # num_fueatures = layer_shape[1:4].num_elements()
        num_fueatures = (int(img_size/4) ** 2) * num_filters2 #this part is hard coded because we 
        #have two max pooling layers that reduce the size of the image to img_size/4.
        layer_flat = tf.reshape(layer,[-1,num_fueatures]) 
        return layer_flat, num_fueatures

    def new_fc_layer(input,num_inputs, num_outputs, use_relu = True):
        weights = new_weights(shape = [num_inputs, num_outputs])
        bias = new_bias(length = num_outputs)
        layer = tf.matmul(input, weights) + bias
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer,weights,bias

    def get_train_model(x):
        #x_image = tf.reshape(x, [-1, img_size, img_size, num_input1]) 
        x_image = tf.expand_dims(x, 3)
        layer_conv1, weight_conv11 = new_convolution_layer(input = x_image, num_input = num_input1, filter_size1 = filter_size, filter_size2 = filter_size, num_output = num_filters1,use_pooling = True, padding = 'SAME')
        layer_conv2, weight_conv12 = new_convolution_layer(input = layer_conv1, num_input = num_filters1,  filter_size1 = filter_size, filter_size2 = filter_size, num_output = num_filters2, use_pooling = True, padding = 'SAME')
        flatten_layer_conv2, size_of_flattened_conv2 = flatten_layer(layer_conv2)
        #first fc layer, training model can only take images of fixed size 
        weights_fc1 = new_weights(shape = [size_of_flattened_conv2, num_fc1])
        bias_fc1 = new_bias(length = num_fc1)
        layer_fc1 = tf.matmul(flatten_layer_conv2, weights_fc1) + bias_fc1
        layer_fc1 = tf.nn.relu(layer_fc1)
        #second fc layer-------------------------------------------------------------------
        weights_fc2 = new_weights(shape = [num_fc1, num_classification])
        bias_fc2 = new_bias(length = num_classification)
        layer_fc2 = tf.matmul(layer_fc1, weights_fc2) + bias_fc2
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= layer_fc2, labels=y)
        cost = tf.reduce_mean(cross_entropy) + (tf.nn.l2_loss(bias_fc1) + tf.nn.l2_loss(weights_fc2) + tf.nn.l2_loss(weights_fc1) + tf.nn.l2_loss(bias_fc1)) * reg_facor
        return layer_fc2,cost

    def get_detection_model(x):
        x_image = tf.expand_dims(x, 3) 
        layer_conv1, weight_conv11 = new_convolution_layer(input = x_image, num_input = num_input1, filter_size1 = filter_size, filter_size2 = filter_size, num_output = num_filters1,use_pooling = True, padding = 'SAME')
        layer_conv2, weight_conv12 = new_convolution_layer(input = layer_conv1, num_input = num_filters1,  filter_size1 = filter_size, filter_size2 = filter_size, num_output = num_filters2, use_pooling = True, padding = 'SAME')
        flatten_layer_conv2, size_of_flattened_conv2 = flatten_layer(layer_conv2)
        #first fc layer, here the fully connected layers are converted to conv layers so the network can 'slide' on images
        #this is the only difference between training model that takes images of fixed size and detection model that can
        #handle images of any sizes 
        weights_fc1 = new_weights(shape = [size_of_flattened_conv2, num_fc1])
        weights_conv3 = tf.reshape(weights_fc1, [int(image_size/4), int(image_size/4), num_filters2, num_fc1])
        bias_fc1 = new_bias(length = num_fc1)
        layer_conv3 = tf.nn.conv2d(input = layer_conv2,  filter = weights_conv3, strides = [1,2,2,1],  padding = 'VALID') + bias_fc1
        layer_conv3 = tf.nn.relu(layer_conv3)
        #second fc layer, here the fully connected layers are converted to conv layers so the network can 'slide' on images
        #this is the only difference between training model that takes images of fixed size and detection model that can
        #handle images of any sizes 
        weights_fc2 = new_weights(shape = [num_fc1, num_classification])
        weights_conv4 = tf.reshape(weights_fc2, [1, 1, num_fc1, num_classification])
        bias_fc2 = new_bias(length = num_classification)
        layer_conv4 = tf.nn.conv2d(input = layer_conv3,  filter = weights_conv4, strides = [1,1,1,1], padding='VALID') + bias_fc2
        #catergory = tf.argmax(layer_conv4)
        return layer_conv4

    def getData(dirs, numOfClass, start, batch_size,shuffle = False):
        trainX, trainY = [], []
        placeholder = [0 for x in range(numOfClass)]
        for i in range(numOfClass):
            current_dir = dirs[i]
            imageNames = (os.listdir(current_dir))
            for j in range(start, start + batch_size):
                imageName = imageNames[j]
                imageName = os.path.join(current_dir, imageName)
                img = cv2.imread(imageName)
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgGrayFilter = cv2.GaussianBlur(imgGray, (3,3), 0)
                imgGrayFilterResize = cv2.resize(imgGrayFilter,(image_size, image_size))
                trainX.append(imgGrayFilterResize)
                placeholder_temp = np.copy(placeholder)
                placeholder_temp[i] = 1
                trainY.append(placeholder_temp)
        return trainX, trainY

    def train_neural_network(num_iterations, dirs):
        y_true = tf.placeholder(tf.float32,shape = [None, num_classification], name = 'y_true')
        y_true_cls = tf.argmax(y_true, dimension = 1)
        raw_prediction, cost = get_train_model(x)
        y_pred = tf.nn.softmax(raw_prediction[0])
        y_pred_cls = tf.argmax(y_pred, dimension = 1)
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = raw_prediction, labels = y)
        #cost = tf.reduce_mean(cross_entropy) + tf.nn.l2_loss()
        if resume == 1:
            optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)
            #optimizer = tf.train.AdamOptimizer().minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        fileToSave = os.path.join(dataDir, "StopSignModel.ckpt")
        error = 1000
        with tf.Session() as sess:
            start = 0
            global resume
            sess.run(tf.global_variables_initializer())
            if resume == 1:
                saver.restore(sess, fileToSave)
            poslen = len(os.listdir(dirs[0]))
            for i in range(num_iterations):
                total_cost = 0
                start = 0
                max_num_of_file = 13000
                while start < max_num_of_file - batch_size:
                    x_train, y_train = getData(dirs, num_classification, start, batch_size)
                    feed_dict_train = {x: x_train, y: y_train}
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
