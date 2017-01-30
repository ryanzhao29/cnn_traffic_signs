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
global_x = tf.placeholder(tf.float32, shape = [None, None, None, color_channels])
global_y = tf.placeholder(tf.float32, shape = [None, classification_num])
image_dir = r'C:\Users\user\Desktop\test\traffic sign detection data2\\'
data_dir = r'C:\Users\user\Desktop\test\checkpoint\\'
#dirs = [str(ii) for ii in range(classification_num)]
classification_output_placeholder = [0 for x in range(classification_num)]