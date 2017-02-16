#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import fcn32_vgg_train
import utils

from tensorflow.python.framework import ops

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

#img1 = skimage.io.imread("./test_data/tabby_cat.png")
img1 = skimage.io.imread("./test_data/19.jpg")
lbl1 = skimage.io.imread("./test_data/19_object.png")

FLAGS = tf.app.flags.FLAGS
with tf.Session() as sess:
    images = tf.placeholder("float")
    labels = tf.placeholder("float")
    feed_dict = {images: img1, labels: lbl1[:,:,0]}
    
    batch_images = tf.expand_dims(images, 0)    
    batch_labels = tf.expand_dims(labels, 0)


    vgg_fcn = fcn32_vgg_train.FCN32VGG()
    
    with tf.name_scope("content_vgg"):
        #vgg_fcn.build(batch_images, debug=True)
        
        vgg_fcn.build(batch_images, train=True, num_classes=1, random_init_fc8=True)

    print('Finished building Network.')

    
    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')   
    
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up, batch_labels]
    down, up, labels_get = sess.run(tensors, feed_dict=feed_dict)
    
    logits = vgg_fcn.pred_up
    labels = batch_labels
    
    loss = sess.run(vgg_fcn.loss_study(logits, labels, 1, head=None), feed_dict=feed_dict)
    
    epsilon_get = sess.run(vgg_fcn.epsilon , feed_dict=feed_dict) 
    
    print(epsilon_get)