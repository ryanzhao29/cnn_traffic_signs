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

    def get_training_model(self, x):
        conv1, weight1 = new_convolution_layer(x, self.color_channel, self.filter_size_conv_layer1, self.filter_size_conv_layer1, self.num_filter_conv_layer1, True, 'SAME')
        conv2, weight2 = new_convolution_layer(conv1, self.num_filter_conv_layer1, self.filter_size_conv_layer2, self.filter_size_conv_layer2, self.num_filter_conv_layer2, True, 'SAME')
        fltten_layer, num_fueatures = flatten_layer(conv2, self.num_filter_conv_layer2, self.image_size)
        fc1,weight_fc1,bias_fc1 = new_fc_layer(fltten_layer,num_fueatures, self.num_fc_layer1, use_relu = True)
        fc2,weight_fc2,bias_fc2 = new_fc_layer(fc1,self.num_fc_layer1, self.num_fc_layer2, use_relu = True)
        fc3,weight_fc3,bias_fc3 = new_fc_layer(fc2,self.num_fc_layer2, self.num_fc_layer3, use_relu = False)
        return fc3


leNet = def_leNet()
leNet.get_training_model(x)

