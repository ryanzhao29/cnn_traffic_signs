import tensorflow as tf
import numpy as np
import os
import cv2
import random
import time
from sklearn.utils import shuffle
image_size = 32
classification_num = 44
classification_num_usa = 6
color_channels = 1
dropout_keep_prob = 0.5
reg_facor = 0.01
batch_size = 20
data_augmentation_factor = 5 #for each sample generate number of data_augmentation_factor new samples by randomly change the original sample
global_x = tf.placeholder(tf.float32, shape = [None, None, None, color_channels])
global_y = tf.placeholder(tf.float32, shape = [None, classification_num])
global_y_FE = tf.placeholder(tf.float32, shape = [None, classification_num_usa])
image_dir = r'C:\Users\user\Desktop\test\traffic sign detection data2\\'
image_dir_usa = r'C:\Users\user\Desktop\test\speed limit and traffic sign\\'
data_dir = r'C:\Users\user\Desktop\test\checkpoint'

class class_feature_extraction_status():
    def __init__(self):
        self.bottleneck          = 0 #means train the original network, to provide the 'pretrained' network
        self.feature_extraction  = 1 #train the new fc layer only
        self.fine_tune           = 2 #train both the new fc layer and the network that provide bottleneck feature
feature_extraction_status = class_feature_extraction_status()

traffic_sign_dictionary = [ '0 = speed limit 20 (prohibitory)',
                            '1 = speed limit 30 (prohibitory)',
                            '2 = speed limit 50 (prohibitory)',
                            '3 = speed limit 60 (prohibitory)',
                            '4 = speed limit 70 (prohibitory)',
                            '5 = speed limit 80 (prohibitory)',
                            '6 = restriction ends 80 (other)',
                            '7 = speed limit 100 (prohibitory)',
                            '8 = speed limit 120 (prohibitory)',
                            '9 = no overtaking (prohibitory)',
                            '10 = no overtaking (trucks) (prohibitory)',
                            '11 = priority at next intersection (danger)',
                            '12 = priority road (other)',
                            '13 = give way (other)',
                            '14 = stop (other)',
                            '15 = no traffic both ways (prohibitory)',
                            '16 = no trucks (prohibitory)',
                            '17 = no entry (other)',
                            '18 = danger (danger)',
                            '19 = bend left (danger)',
                            '20 = bend right (danger)',
                            '21 = bend (danger)',
                            '22 = uneven road (danger)',
                            '23 = slippery road (danger)',
                            '24 = road narrows (danger)',
                            '25 = construction (danger)',
                            '26 = traffic signal (danger)',
                            '27 = pedestrian crossing (danger)',
                            '28 = school crossing (danger)',
                            '29 = cycles crossing (danger)',
                            '30 = snow (danger)',
                            '31 = animals (danger)',
                            '32 = restriction ends (other)',
                            '33 = go right (mandatory)',
                            '34 = go left (mandatory)',
                            '35 = go straight (mandatory)',
                            '36 = go right or straight (mandatory)',
                            '37 = go left or straight (mandatory)',
                            '38 = keep right (mandatory)',
                            '39 = keep left (mandatory)',
                            '40 = roundabout (mandatory)',
                            '41 = restriction ends (overtaking) (other)',
                            '42 = restriction ends (overtaking (trucks)) (other)',
                            '43 = background (no traffic sign)']

traffic_sign_dictionary_US = ['0 = background',
                              '1 = stop sign',
                              '2 = speed limit 25 mph',
                              '3 = speed limit 35 mph',
                              '4 = speed limit 45 mph',
                              '5 = speed limit 55 mph',
                              '6 = speed limit 65 mph',
                              '7 = merge right',
                              '8 = merge left',
                              '9 = pedestrian crossing',
                              '10 = right turn only',
                              '11 = no pass',
                              '12 = pass with care',
                              '13 = round about',
                              '14 = no turn on red',
                              '15 = speed limit 75 mph']