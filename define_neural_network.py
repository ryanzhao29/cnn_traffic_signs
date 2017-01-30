from global_head_file import *

def new_weights(shape):
    return tf.Variable(tf.random_normal(shape, 0, 0.05))

def new_bias(length):
    return tf.Variable(tf.random_normal(shape = [length]))

def new_convolution_layer(input, num_input, filter_size1, filter_size2, num_output, use_pooling = True, padding = 'SAME'):
    #conv 层的的权重的纬度[卷积滤波器宽度, 卷积滤波器宽度, 本层输入数量, 本层输出数量]
    shape = [filter_size1, filter_size2, num_input, num_output] #[filter_size1, filter_size2, num_input,num_output]
    weights = new_weights(shape)
    bias = new_bias(num_output)
    layer = tf.nn.conv2d(input = input, filter = weights, strides = [1,1,1,1],padding = padding) + bias
    if use_pooling:
        layer = tf.nn.max_pool(value = layer, ksize = [1,2,2,1],strides = [1,2,2,1],padding = padding)
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer, num_filters2, img_size):
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

def get_train_model(x): #this is an example of a simple cnn for training purpose, not used anywhere in the code
    #x_image = tf.reshape(x, [-1, img_size, img_size, num_input1]) 
    layer_conv1, weight_conv11 = new_convolution_layer(input = x, num_input = num_input1, filter_size1 = filter_size, filter_size2 = filter_size, num_output = num_filters1,use_pooling = True, padding = 'SAME')
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

def get_detection_model(x):  #this is an example of a simple cnn for detection purpose, not use anywhere in the code
    layer_conv1, weight_conv11 = new_convolution_layer(input = x, num_input = num_input1, filter_size1 = filter_size, filter_size2 = filter_size, num_output = num_filters1,use_pooling = True, padding = 'SAME')
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

