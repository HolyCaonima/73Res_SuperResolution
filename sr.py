
import math
import customDataGeter
import model
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
import prettytensor as pt

from scipy.misc import imsave, imshow, imresize

class SR(object):

    def __init__(self, version, batch_size, learning_rate, data_directory, valid_directory, log_directory):
        self.img_size = [64, 64]
        self.version = version
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_directory = data_directory
        self.valid_directory = valid_directory
        self.log_directory = log_directory

        # build the graph
        self._build_graph()
        self.merged_all = tf.summary.merge_all()
       
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.log_directory, self.sess.graph)


    def _build_graph(self):
        self.valid_input = customDataGeter.input(self.valid_directory, self.img_size, self.batch_size)
        self.gt_input = customDataGeter.input(self.data_directory, self.img_size, self.batch_size)
        self.feed_into = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.feed_gt = tf.placeholder(tf.float32, shape=[None, None, None, 3])

        img_out = ly.conv2d(self.feed_into, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 32, 3)
        img_out = model.build_block_fukc(img_out)

        img_out = ly.conv2d(img_out, 3, 3, activation_fn=tf.nn.tanh)

        self.sr_out = img_out + self.feed_into

        # construct loss___ pixel wise difference
        diff = self.sr_out-self.feed_gt
        self.loss = tf.reduce_sum(tf.abs(diff))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # construct queue low res
        self.queue_low = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 3])
        self.queue_low_out = tf.image.resize_bilinear(self.queue_low, [self.img_size[0]/4,self.img_size[1]/4])
        self.queue_low_out = tf.image.resize_bilinear(self.queue_low_out, [self.img_size[0],self.img_size[1]])

    def _gt_two_part(self, input_img):
        input_gts = self.sess.run(input_img)
        input_low = np.array(input_gts)*255
        for i in range(input_low.shape[0]):
            input_low[i] = imresize(imresize(input_low[i], [self.img_size[0]/4, self.img_size[0]/4]), [self.img_size[0], self.img_size[0]])/255.0
        #input_low = self.sess.run(self.queue_low_out,feed_dict={self.queue_low:input_gts})
        return input_gts, input_low

    def train(self):
        input_gts, input_low = self._gt_two_part(self.gt_input)
        self.sess.run(self.optimizer,feed_dict={self.feed_into:input_low, self.feed_gt:input_gts})
    
    def test_single(self, img):
        img = np.reshape(img, (1, img.shape[0], img.shape[1],3))
        result = self.sess.run(self.sr_out, feed_dict={self.feed_into:img})
        return result[0]

    def get_loss(self):
        input_gts, input_low = self._gt_two_part(self.gt_input)
        true_loss = self.sess.run(self.loss, feed_dict={self.feed_into:input_low, self.feed_gt:input_gts})
        gt_low = np.sum(np.abs(input_gts-input_low))
        return true_loss, gt_low, gt_low-true_loss

    def generate_and_save_images(self, directory):
        input_gts, input_low = self._gt_two_part(self.valid_input)
        resolution = self.sess.run(self.sr_out,feed_dict={self.feed_into:input_low})
        if not os.path.exists(directory+'/img'):
            os.mkdir(directory+'/img')
        for i in range(input_gts.shape[0]):
            out = np.concatenate((input_gts[i],resolution[i],input_low[i],np.abs(resolution[i]-input_low[i]),np.abs(input_gts[i]-input_low[i])),1)
            imsave(directory+'/img/'+str(i)+'.jpg',out)
        
        input_gts, input_low = self._gt_two_part(self.gt_input)
        resolution = self.sess.run(self.sr_out,feed_dict={self.feed_into:input_low})
        for i in range(input_gts.shape[0]):
            out = np.concatenate((input_gts[i],resolution[i],input_low[i],np.abs(resolution[i]-input_low[i]),np.abs(input_gts[i]-input_low[i])),1)
            imsave(directory+'/img/tri_'+str(i)+'.jpg',out)

    
    def get_merged_image(self, num_samples):
        pass