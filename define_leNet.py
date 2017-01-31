#this file is a realization of the lenet propsed by lecuun
#the training mode is the default model of lenet
#whereas the detection model is full convolutional net where the fc layers are replaced by conv layers so the network can take images of any sizes. 
from define_neural_network import *
from global_head_file import *
class def_leNet():
    def __init__(self):
        self.mu = 0
        self.color_channel = color_channels #3 for color images 1 for grey scale images
        self.sigma = 0.1
        self.image_size = image_size
        self.filter_size_conv_layer1 = 5
        self.num_filter_conv_layer1 = 6
        self.use_maxpooling_conv_layer1 = True
        self.filter_size_conv_layer2 = 5
        self.num_filter_conv_layer2 = 16
        self.use_maxpooling_conv_layer2 = True
        self.num_fc_layer1 = 120
        self.num_fc_layer2 = 84
        self.num_fc_layer3 = classification_num
        #self.weights_conv1
        #self.weights_conv2
        #self.weights_fc1
        #self.weights_fc2
        #self.weights_fc3
        #self.bias_conv1
        #self.bias_conv2
        #self.bias_fc1
        #self.bias_fc2
        #self.bias_fc3

    def get_training_model(self, enable_dropout = True):
        conv1, self.weights_conv1, self.bias_conv1 = new_convolution_layer(global_x, self.color_channel, self.filter_size_conv_layer1, self.filter_size_conv_layer1, self.num_filter_conv_layer1, True, 'SAME')
        conv2, self.weights_conv2, self.bias_conv2 = new_convolution_layer(conv1, self.num_filter_conv_layer1, self.filter_size_conv_layer2, self.filter_size_conv_layer2, self.num_filter_conv_layer2, True, 'SAME')
        fltten_layer, num_fueatures = flatten_layer(conv2, self.num_filter_conv_layer2, self.image_size)
        fc1,self.weights_fc1,self.bias_fc1 = new_fc_layer(fltten_layer, num_fueatures, self.num_fc_layer1, use_relu = True)
        fc2,self.weights_fc2,self.bias_fc2 = new_fc_layer(fc1, self.num_fc_layer1, self.num_fc_layer2, use_relu = True)
        #dropout probability = 0.5
        if enable_dropout == True:
            fc2 = tf.nn.dropout(fc2, keep_prob = dropout_keep_prob)
        fc3,self.weights_fc3,self.bias_fc3 = new_fc_layer(fc2, self.num_fc_layer2, self.num_fc_layer3, use_relu = False)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = global_y)
        # regulization 
        cost =     tf.reduce_mean(cross_entropy) +\
                (  tf.nn.l2_loss(self.weights_fc1) + tf.nn.l2_loss(self.bias_fc1) \
                 + tf.nn.l2_loss(self.weights_fc2) + tf.nn.l2_loss(self.bias_fc2) \
                 + tf.nn.l2_loss(self.weights_fc3) + tf.nn.l2_loss(self.bias_fc3))  * reg_facor
        return fc2, fc3, cost #herer we return the fc2 for transfer learning purpose

    def get_detection_model(self):
        conv1, self.weights_conv1, self.bias_conv1  = new_convolution_layer(global_x, self.color_channel, self.filter_size_conv_layer1, self.filter_size_conv_layer1, self.num_filter_conv_layer1, True, 'SAME')
        conv2, self.weights_conv2, self.bias_conv2 = new_convolution_layer(conv1, self.num_filter_conv_layer1, self.filter_size_conv_layer2, self.filter_size_conv_layer2, self.num_filter_conv_layer2, True, 'SAME')
        fltten_layer, num_fueatures = flatten_layer(conv2, self.num_filter_conv_layer2, self.image_size)
        #need to conver the 3 dense layer to conv layer so the network can take images of any size
        #fc1, weights_fc1, bias_fc1 = new_fc_layer(fltten_layer, num_fueatures, self.num_fc_layer1, use_relu = True)
        #here we have to manually transform the dense layer to conv layer as our function dose not take weights and bias as inputs
        #this is the original weights and bias of fc1
        self.weights_fc1 = new_weights(shape = [num_fueatures, self.num_fc_layer1])
        self.bias_fc1 = new_bias(self.num_fc_layer1)
        #transform this weight and bias for conv layer
        weights_conv_fc1 = tf.reshape(self.weights_fc1, [int(image_size/4), int(image_size/4), self.num_filter_conv_layer2, self.num_fc_layer1])
        layer_conv_fc1 = tf.nn.conv2d(conv2, weights_conv_fc1, [1,1,1,1], 'VALID') + self.bias_fc1
        layer_conv_fc1 = tf.nn.relu(layer_conv_fc1)
        #repeat this for fc2
        self.weights_fc2 = new_weights(shape = [self.num_fc_layer1, self.num_fc_layer2])
        self.bias_fc2 = new_bias(self.num_fc_layer2)
        #transform this weight and bias for conv layer
        weights_conv_fc2 = tf.reshape(self.weights_fc2, [1, 1 , self.num_fc_layer1, self.num_fc_layer2])
        layer_conv_fc2 = tf.nn.conv2d(layer_conv_fc1, weights_conv_fc2, [1,1,1,1], 'VALID') + self.bias_fc2
        layer_conv_fc2 = tf.nn.relu(layer_conv_fc2)
        #repeat this for fc3
        self.weights_fc3 = new_weights(shape = [self.num_fc_layer2, self.num_fc_layer3])
        self.bias_fc3 = new_bias(self.num_fc_layer3)
        #transform this weight and bias for conv layer
        weights_conv_fc3 = tf.reshape(self.weights_fc3, [1, 1 , self.num_fc_layer2, self.num_fc_layer3])
        layer_conv_fc3 = tf.nn.conv2d(layer_conv_fc2, weights_conv_fc3, [1,1,1,1], 'VALID') + self.bias_fc3
        #layer_conv_fc2 = tf.nn.relu(layer_conv_fc2) final out put dose not use relu
        return layer_conv_fc3

# create an instance of LeNet
leNet = def_leNet()


