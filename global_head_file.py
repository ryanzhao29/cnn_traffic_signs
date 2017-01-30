import tensorflow as tf
image_size = 32
classification_num = 43
color_channels = 3
x = tf.placeholder(tf.float32, shape = [None, None, None, color_channels])
y = tf.placeholder(tf.float32, shape = [None, classification_num])