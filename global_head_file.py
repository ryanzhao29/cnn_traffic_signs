import tensorflow as tf
import numpy as np
import os
import cv2
import random
import time

image_size = 32
classification_num = 43
color_channels = 3
dropout_keep_prob = 0.5
reg_facor = 0.01
batch_size = 20
data_augmentation_factor = 10 #for each sample generate number of data_augmentation_factor new samples by randomly change the original sample
global_x = tf.placeholder(tf.float32, shape = [None, None, None, color_channels])
global_y = tf.placeholder(tf.float32, shape = [None, classification_num])
image_dir = r'C:\Users\user\Desktop\test\traffic sign detection data2\\'
data_dir = r'C:\Users\user\Desktop\test\checkpoint'
#dirs = [str(ii) for ii in range(classification_num)]
classification_output_placeholder = [0 for x in range(classification_num)]

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
                            '42 = restriction ends (overtaking (trucks)) (other)']