# coding=utf-8
"""Implementation of VMI-DI-TI-SI-FGSM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.fftpack import dct, idct, rfft, irfft
import os
import numpy as np
import cv2
import pandas as pd
import scipy.stats as st
from tensorpack import TowerContext
from attack_method import *
from utils import *
from tqdm import tqdm
from imageio import imread, imsave
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate
###################CUDA_VISIBLE_DEVICES=1 python3 mi.py --pre_attack 5#######
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random


slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer('number', 20, 'the number of images for variance tuning')

tf.flags.DEFINE_integer('pre_attack', 5, 'the number of images for variance tuning')

tf.flags.DEFINE_float('beta', 1.5, 'the bound for variance tuning.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_float('b', 999.0, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')
tf.flags.DEFINE_string('input_csv', 'dev_data/val_rs.csv', 'Input directory with images.')
tf.flags.DEFINE_string('input_dir', 'dev_data/val_rs/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './mioutput_ours',
                       'Output directory with images.')
tf.flags.DEFINE_string('output_dir_noise', './mioutput_ours_noise',
                       'Output directory with images.')

tf.flags.DEFINE_float('last_b', 0.0, 'flooding.')

FLAGS = tf.flags.FLAGS
###################CUDA_VISIBLE_DEVICES=3 python3 mi.py --output_dir mi --pre_attack 0#######
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
###################CUDA_VISIBLE_DEVICES=1 python3 patch-wise_iter_attack.py --b 999##########
model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_101': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
    'vgg_16': os.path.join(FLAGS.checkpoint_path, 'vgg_16.ckpt'),
    'resnet_v2_152': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_152.ckpt'),
}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
P_kern, kern_size = project_kern(3)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def graph(x, x_ori, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    x_ori = tf.cond(tf.equal(i, 0), lambda: x, lambda: x_ori)
    pre_attack = FLAGS.pre_attack
    # NI-FGSM https://arxiv.org/pdf/1908.06281.pdf
    x_nes = x + momentum * alpha * tf.sign(grad)
    
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
           x_nes , num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    pred = tf.argmax(end_points_v3['Predictions'], 1)

    # with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    #     logits_v4, end_points_v4 = inception_v4.inception_v4(
    #         x_nes, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    # pred = tf.argmax(end_points_v4['Predictions'], 1)

    # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    #     logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
    #         x_nes, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    # pred = tf.argmax(logits_resnet, 1)

    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    #     logits_Incres, end_points_IR = inception_resnet_v2.inception_resnet_v2(
    #         x_nes, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    # pred = tf.argmax(logits_Incres, 1)

    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    noise = tf.gradients(cross_entropy, x)[0]

    
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    
    x = tf.cond(tf.equal(i, pre_attack), lambda: x_ori, lambda: x)
    amp = tf.cond(tf.less(i, pre_attack), lambda: 10.0, lambda: 1.0)
    x = x + amp * alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, x_ori, y, i, x_max, x_min, noise


def stop(x, x_ori, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter + FLAGS.pre_attack
    return tf.less(i, num_iter)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    sum_ensadv_res_v2 = 0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        adv_img = tf.placeholder(tf.float32, shape=batch_shape)

        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = resnet_v2.resnet_v2_101(
        #         adv_img, num_classes=num_classes, is_training=False)
        # pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_v3.inception_v3(
        #             adv_img, num_classes = num_classes, is_training = False, scope = 'AdvInceptionV3')
        # pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        #         adv_img, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
        # pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_v3.inception_v3(
        #         adv_img, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        # pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        #         adv_img, num_classes=num_classes, is_training=False)
        # pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False)
        pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        # with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_v4.inception_v4(
        #         adv_img, num_classes=num_classes, is_training=False)
        # pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_ori = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _,  = tf.while_loop(stop, graph, [x_input, x_ori, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s7.restore(sess, model_checkpoint_map['resnet_v2_101'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2_152'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])
            dev = pd.read_csv(FLAGS.input_csv)
            idx = 0
            l2_diff = 0
            # Attack Period
            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
                save_images(adv_images-images, filenames, FLAGS.output_dir_noise)
                # diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                # l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))
                pre_ensadv_res_v2_ = sess.run(pre_ensadv_res_v2,
                                              feed_dict={adv_img: (adv_images)})
                sum_ensadv_res_v2 += (pre_ensadv_res_v2_ != True_label).sum()
                print(sum_ensadv_res_v2)
                
            # Inference Period    
            # for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
            #     images, filenames, True_label = load_images(FLAGS.output_dir, dev, idx * FLAGS.batch_size, batch_shape)
            #     images = images.astype(np.float32)
            #     pre_ensadv_res_v2_ = sess.run(pre_ensadv_res_v2,
            #                             feed_dict = {adv_img: (images)}) 
            #     sum_ensadv_res_v2 += (pre_ensadv_res_v2_ != True_label).sum()
            #     print(sum_ensadv_res_v2)


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
