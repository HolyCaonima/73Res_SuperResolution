

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf

from tensorflow.contrib import layers, losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave, imshow, imresize, imread

from tensorflow.examples.tutorials.mnist import input_data

from progressbar import ProgressBar

from sr import SR

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("global_step", 0, "the step of current training")
flags.DEFINE_integer("batch_size", 32, "bathch size")
flags.DEFINE_integer("updates_per_epoch", 100, "update certain times then show the loss")
flags.DEFINE_integer("max_epoch", 500, "max epoch")
flags.DEFINE_string("working_directory", "./work", "the working directory of current job")
flags.DEFINE_string("data_directory", "../CutFace/saveTri", "directory of training data")
flags.DEFINE_string("valid_directory", "../CutFace/saveTes", "directory of test data")
flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
flags.DEFINE_integer("version", 1, "the version of model")

flags.DEFINE_string("mode", 'train', "the mode of program")
flags.DEFINE_string("image", './', "the url of test image")

FLAGS = flags.FLAGS

start_epoch = 0

def save_model(sess, saver):
    if not os.path.exists(os.path.join(FLAGS.working_directory,"save")):
        os.mkdir(os.path.join(FLAGS.working_directory,"save"))  
    if os.path.exists(os.path.join(FLAGS.working_directory,"save","desc")):
        os.remove(os.path.join(FLAGS.working_directory,"save","desc"))
    model_desc = open(os.path.join(FLAGS.working_directory,"save","desc"),'w')
    model_desc.write(str(FLAGS.global_step)+"\n")
    model_desc.write(str(FLAGS.batch_size)+"\n")
    model_desc.write(str(FLAGS.updates_per_epoch)+"\n")
    model_desc.write(str(FLAGS.max_epoch)+"\n")
    model_desc.write(str(FLAGS.data_directory)+"\n")
    model_desc.write(str(FLAGS.valid_directory)+"\n")
    model_desc.write(str(FLAGS.learning_rate)+"\n")
    model_desc.write(str(start_epoch)+"\n")
    saver.save(sess, os.path.join(FLAGS.working_directory,"save","model.data"))
    model_desc.close()
    print "model saved!"

def load_desc():
    if not os.path.exists(os.path.join(FLAGS.working_directory,"save", "desc")):
        print "model not exists!"
        return
    model_desc = open(os.path.join(FLAGS.working_directory,"save","desc"))
    FLAGS.global_step = int(model_desc.readline())
    FLAGS.batch_size = int(model_desc.readline())
    FLAGS.updates_per_epoch = int(model_desc.readline())
    FLAGS.max_epoch = int(model_desc.readline())
    FLAGS.data_directory = str(model_desc.readline()).replace("\n","")
    FLAGS.valid_directory = str(model_desc.readline()).replace("\n","")
    FLAGS.learning_rate = float(model_desc.readline())
    start_epoch = int(model_desc.readline())
    model_desc.close()

def load_model(sess, saver):
    if not os.path.exists(os.path.join(FLAGS.working_directory,"save", "desc")):
        print "model not exists!"
        return
    saver.restore(sess, os.path.join(FLAGS.working_directory,"save","model.data"))

def main():
    if not os.path.exists(FLAGS.working_directory):
        os.makedirs(FLAGS.working_directory)
    if not os.path.exists(FLAGS.data_directory):
        os.makedirs(FLAGS.data_directory)

    if os.path.exists(os.path.join(FLAGS.working_directory,"save","ver")):
        ver_desc = open(os.path.join(FLAGS.working_directory,"save","ver"))
        FLAGS.version = int(ver_desc.readline())
        ver_desc.close()
    else:
        if not os.path.exists(os.path.join(FLAGS.working_directory,"save")):
            os.mkdir(os.path.join(FLAGS.working_directory,"save"))
        ver_desc = open(os.path.join(FLAGS.working_directory,"save","ver"),'w')
        ver_desc.write(str(FLAGS.version)+"\n")
        ver_desc.close()

    load_desc()
    sr = SR(FLAGS.version, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.data_directory, FLAGS.valid_directory, os.path.join(FLAGS.working_directory, "log"))
    saver = tf.train.Saver()
    load_model(sr.sess, saver)

    # start the queue runners    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sr.sess, coord=coord)
    if FLAGS.mode=='train':
        for epoch in range(start_epoch, FLAGS.max_epoch):
            pbar = ProgressBar()
            for update in pbar(range(FLAGS.updates_per_epoch)):
                sr.train()
                FLAGS.global_step = FLAGS.global_step + 1
            
            t_l, d_l, f_l = sr.get_loss()
            print "model_loss: " + str(t_l) +"  img_loss: " + str(d_l)+"  diff_loss: "+str(f_l)
            sr.generate_and_save_images(FLAGS.working_directory)
            save_model(sr.sess, saver)

    else:
        test_img = imread(FLAGS.image)/255.0
        result = sr.test_single(test_img)
        imsave(FLAGS.image[0:len(FLAGS.image)-4]+'_sr'+'.'+'jpg',result)

    # ask threads to stop
    coord.request_stop()

    # wait for threads to finish
    coord.join(threads)
    sr.sess.close()
    
if __name__ == '__main__':
    main()