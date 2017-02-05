#this file is a realization of the lenet propsed by lecuun
#the training mode is the default model of lenet
#whereas the detection model is full convolutional net where the fc layers are replaced by conv layers so the network can take images of any sizes. 
from define_neural_network import *
from global_head_file import *
class def_leNet():
    def __init__(self):
        self.mu = 0
        self.sigma = 0.1
        self.color_channels = color_channels #3 for color images 1 for grey scale images
        self.image_size = image_size
        self.filter_size_conv1 = 5
        self.output_num_conv1 = 6
        self.use_maxpooling_conv_layer1 = True
        self.filter_size_conv2 = 5
        self.output_num_conv2 = 16
        self.use_maxpooling_conv_layer2 = True
        self.num_fc_layer1 = 120
        self.num_fc_layer2 = 84
        self.num_fc_layer3 = classification_num
        self.num_fc_layer3_FE = classification_num_usa
        #shape = [filter_size1, filter_size2, num_input, num_output]
        self.weights_conv1 = new_weights(shape = [self.filter_size_conv1, self.filter_size_conv1, self.color_channels, self.output_num_conv1])
        self.weights_conv2 = new_weights(shape = [self.filter_size_conv2, self.filter_size_conv2, self.output_num_conv1, self.output_num_conv2])
        self.weights_fc1 = new_weights(shape = [int(image_size/4) **2 * self.output_num_conv2, self.num_fc_layer1]) #the first dimision should be the same to number of features fron conv2
        self.weights_fc2 = new_weights(shape = [self.num_fc_layer1, self.num_fc_layer2])
        self.weights_fc3 = new_weights(shape= [self.num_fc_layer2, self.num_fc_layer3])
        self.bias_conv1 = new_bias(self.output_num_conv1)
        self.bias_conv2 = new_bias(self.output_num_conv2)
        self.bias_fc1 = new_bias(self.num_fc_layer1)
        self.bias_fc2 = new_bias(self.num_fc_layer2)
        self.bias_fc3 = new_bias(self.num_fc_layer3)
        #feature extraction layer, FE = feature extraction
        self.weights_fc3_FE = new_weights([self.num_fc_layer2, self.num_fc_layer3_FE])
        self.bias_fc3_FE = new_bias(self.num_fc_layer3_FE)

    def get_training_model(self, enable_dropout = True, feature_extraction_phase = feature_extraction_status.bottleneck):

        if feature_extraction_phase == feature_extraction_status.bottleneck: #only the bottleneck training uses the fc layer of the original lenet as the output
            conv1 = new_convolution_layer(global_x, self.weights_conv1, self.bias_conv1, True, 'SAME')
            conv2 = new_convolution_layer(conv1, self.weights_conv2, self.bias_conv2, True, 'SAME')
            fltten_layer, num_fueatures = flatten_layer(conv2, self.output_num_conv2, self.image_size)
            fc1 = new_fc_layer(fltten_layer, self.weights_fc1, self.bias_fc1, use_relu = True)
            if enable_dropout == True:
                fc1 = tf.nn.dropout(fc1, keep_prob = dropout_keep_prob)
            fc2 = new_fc_layer(fc1, self.weights_fc2, self.bias_fc2, use_relu = True)
            # if enable_dropout == True:# dropout probability = 0.5
            #     fc2 = tf.nn.dropout(fc2, keep_prob = dropout_keep_prob)
            fc3 = new_fc_layer(fc2, self.weights_fc3, self.bias_fc3, use_relu = False)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = global_y)
            cost =  (  tf.nn.l2_loss(self.weights_fc1) + tf.nn.l2_loss(self.bias_fc1)               \
                     + tf.nn.l2_loss(self.weights_fc2) + tf.nn.l2_loss(self.bias_fc2)               \
                     + tf.nn.l2_loss(self.weights_fc3) + tf.nn.l2_loss(self.bias_fc3)) * reg_facor  \
                     + tf.reduce_mean(cross_entropy) # run regulization on all lenet fc layers

        elif feature_extraction_phase == feature_extraction_status.feature_extraction: #feature extraction use new fc layer and run regulization on new fc weights only
            conv1 = new_convolution_layer(global_x, self.weights_conv1, self.bias_conv1, True, 'SAME')
            conv2 = new_convolution_layer(conv1, self.weights_conv2, self.bias_conv2, True, 'SAME')
            fltten_layer, num_fueatures = flatten_layer(conv2, self.output_num_conv2, self.image_size)
            fltten_layer = tf.stop_gradient(fltten_layer)  # stop gradient so only optimze bias_fc3_FE and weights_fc3_FE
            fc1 = new_fc_layer(fltten_layer, self.weights_fc1, self.bias_fc1, use_relu=True)
            if enable_dropout == True:
                fc1 = tf.nn.dropout(fc1, keep_prob=dropout_keep_prob)
            fc2 = new_fc_layer(fc1, self.weights_fc2, self.bias_fc2, use_relu=True)
            # if enable_dropout == True:            # dropout probability = 0.5
            #     fc2 = tf.nn.dropout(fc2, keep_prob=dropout_keep_prob)
            fc3 = new_fc_layer(fc2, self.weights_fc3_FE, self.bias_fc3_FE, use_relu = False) 
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = global_y_FE)
            cost =  (  tf.nn.l2_loss(self.weights_fc1)      + tf.nn.l2_loss(self.bias_fc1)                    \
                     + tf.nn.l2_loss(self.weights_fc2)      + tf.nn.l2_loss(self.bias_fc2)                    \
                     + tf.nn.l2_loss(self.weights_fc3_FE)   + tf.nn.l2_loss(self.bias_fc3_FE)) * reg_facor \
                     + tf.reduce_mean(cross_entropy) # run regulization on all lenet fc layers

        elif feature_extraction_phase == feature_extraction_status.fine_tune: #fine tune use new fc layer and run regulization on new fc weights and old weights of the first two fc layers
            conv1 = new_convolution_layer(global_x, self.weights_conv1, self.bias_conv1, True, 'SAME')
           # conv1 = tf.stop_gradient(conv1)
            conv2 = new_convolution_layer(conv1, self.weights_conv2, self.bias_conv2, True, 'SAME')
            fltten_layer, num_fueatures = flatten_layer(conv2, self.output_num_conv2, self.image_size)
            fc1 = new_fc_layer(fltten_layer, self.weights_fc1, self.bias_fc1, use_relu=True)
            if enable_dropout == True:
                fc1 = tf.nn.dropout(fc1, keep_prob=dropout_keep_prob)
            fc2 = new_fc_layer(fc1, self.weights_fc2, self.bias_fc2, use_relu=True)
            # if enable_dropout == True:# dropout probability = 0.5
            #     fc2 = tf.nn.dropout(fc2, keep_prob=dropout_keep_prob)
            fc3 = new_fc_layer(fc2, self.weights_fc3_FE, self.bias_fc3_FE, use_relu = False)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = global_y_FE)
            cost =  (   tf.nn.l2_loss(self.weights_fc1)     + tf.nn.l2_loss(self.bias_fc1)                  \
                      + tf.nn.l2_loss(self.weights_fc2)     + tf.nn.l2_loss(self.bias_fc2)                  \
                      + tf.nn.l2_loss(self.weights_fc3_FE)  + tf.nn.l2_loss(self.bias_fc3_FE)) * reg_facor  \
                      + tf.reduce_mean(cross_entropy)
        else:#should never go here
            print ('incorrect status has been chosen, check code.')
            return

        return fc2, fc3, cost, tf.reduce_mean(cross_entropy)

    def get_detection_model(self, feature_extraction = False): 
        conv1 = new_convolution_layer(global_x, self.weights_conv1, self.bias_conv1, True, 'SAME')
        conv2 = new_convolution_layer(conv1, self.weights_conv2, self.bias_conv2, True, 'SAME')
        #transform this weight and bias for conv layer
        weights_conv_fc1 = tf.reshape(self.weights_fc1, [int(image_size/4), int(image_size/4), self.output_num_conv2, self.num_fc_layer1])
        layer_conv_fc1 = tf.nn.conv2d(conv2, weights_conv_fc1, [1,1,1,1], 'VALID') + self.bias_fc1
        layer_conv_fc1 = tf.nn.relu(layer_conv_fc1)
        #repeat this for fc2
        #transform this weight and bias for conv layer
        weights_conv_fc2 = tf.reshape(self.weights_fc2, [1, 1 , self.num_fc_layer1, self.num_fc_layer2])
        layer_conv_fc2 = tf.nn.conv2d(layer_conv_fc1, weights_conv_fc2, [1,1,1,1], 'VALID') + self.bias_fc2
        layer_conv_fc2 = tf.nn.relu(layer_conv_fc2)
        #repeat this for fc3 or feature extraction layer
        if feature_extraction == False: #this is the orginal lenet with 44 outputs
            #transform the weight and bias for conv layer
            weights_conv_fc3 = tf.reshape(self.weights_fc3, [1, 1 , self.num_fc_layer2, self.num_fc_layer3])
            layer_conv_fc3 = tf.nn.conv2d(layer_conv_fc2, weights_conv_fc3, [1,1,1,1], 'VALID') + self.bias_fc3
        else: #use the bottleneck feature of the orginal lenet but with a differnt final fc layer(here is conv layer for sliding window purpose)
            weights_conv_fc3_FE = tf.reshape(self.weights_fc3_FE, [1, 1 , self.num_fc_layer2, self.num_fc_layer3_FE])
            layer_conv_fc3 = tf.nn.conv2d(layer_conv_fc2, weights_conv_fc3_FE, [1,1,1,1], 'VALID') + self.bias_fc3_FE
        #layer_conv_fc2 = tf.nn.relu(layer_conv_fc2) final out put dose not use relu
        return layer_conv_fc3

#define an instance of leNet
leNet = def_leNet()

    
    #def get_detection_model_feature_extraction(self):
    #    conv1, self.weights_conv1, self.bias_conv1  = new_convolution_layer(global_x, self.color_channel, self.filter_size_conv1, self.filter_size_conv1, self.output_num_conv1, True, 'SAME')
    #    conv2, self.weights_conv2, self.bias_conv2 = new_convolution_layer(conv1, self.output_num_conv1, self.filter_size_conv2, self.filter_size_conv2, self.output_num_conv2, True, 'SAME')
    #    fltten_layer, num_fueatures = flatten_layer(conv2, self.output_num_conv2, self.image_size)
    #    #need to conver the 3 dense layer to conv layer so the network can take images of any size
    #    #fc1, weights_fc1, bias_fc1 = new_fc_layer(fltten_layer, num_fueatures, self.num_fc_layer1, use_relu = True)
    #    #here we have to manually transform the dense layer to conv layer as our function dose not take weights and bias as inputs
    #    #this is the original weights and bias of fc1
    #    self.weights_fc1 = new_weights(shape = [num_fueatures, self.num_fc_layer1])
    #    self.bias_fc1 = new_bias(self.num_fc_layer1)
    #    #transform this weight and bias for conv layer
    #    weights_conv_fc1 = tf.reshape(self.weights_fc1, [int(image_size/4), int(image_size/4), self.output_num_conv2, self.num_fc_layer1])
    #    layer_conv_fc1 = tf.nn.conv2d(conv2, weights_conv_fc1, [1,1,1,1], 'VALID') + self.bias_fc1
    #    layer_conv_fc1 = tf.nn.relu(layer_conv_fc1)
    #    #repeat this for fc2
    #    self.weights_fc2 = new_weights(shape = [self.num_fc_layer1, self.num_fc_layer2])
    #    self.bias_fc2 = new_bias(self.num_fc_layer2)
    #    #transform this weight and bias for conv layer
    #    weights_conv_fc2 = tf.reshape(self.weights_fc2, [1, 1 , self.num_fc_layer1, self.num_fc_layer2])
    #    layer_conv_fc2 = tf.nn.conv2d(layer_conv_fc1, weights_conv_fc2, [1,1,1,1], 'VALID') + self.bias_fc2
    #    layer_conv_fc2 = tf.nn.relu(layer_conv_fc2)
    #    #repeat this for fc3 but use fc obtained from feature extraction
    #    #these weights_fc3 and self.bias_fc3  are placeholder to pass the saver, they are not used anywhere in the feature_extraction model
    #    self.weights_fc3 = new_weights(shape = [self.num_fc_layer2, self.num_fc_layer3])
    #    self.bias_fc3 = new_bias(self.num_fc_layer3)

    #    self.weights_fc3_FE = new_weights(shape = [self.num_fc_layer2, self.num_fc_layer3_FE])
    #    self.bias_fc3_FE = new_bias(self.num_fc_layer3_FE)
    #    #transform this weight and bias for conv layer
    #    weights_conv_fc3_usa = tf.reshape(self.weights_fc3_FE, [1, 1 , self.num_fc_layer2, self.num_fc_layer3_FE ])
    #    layer_conv_fc3_usa = tf.nn.conv2d(layer_conv_fc2, weights_conv_fc3_usa, [1,1,1,1], 'VALID') + self.bias_fc3_FE
    #    #layer_conv_fc2 = tf.nn.relu(layer_conv_fc2) final out put dose not use relu
    #    return layer_conv_fc3_usa

# create an instance of LeNet


