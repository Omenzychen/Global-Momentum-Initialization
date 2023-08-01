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
###################CUDA_VISIBLE_DEVICES=1 python3 vmi.py --output_dir temp1 --pre_attack 5 #######
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random


slim = tf.contrib.slim

tf.flags.DEFINE_integer(
    'batch_size', 2, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_float('portion', 0.2, 'protion for the mixed image')

tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_integer(
    'size', 3, 'Number of randomly sampled images')

tf.flags.DEFINE_integer(
    'number', 20, 'the number of images for variance tuning')

tf.flags.DEFINE_integer(
    'pre_attack', 5, 'the number of images for variance tuning')

tf.flags.DEFINE_float('beta', 1.5, 'the bound for variance tuning.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')
tf.flags.DEFINE_string('input_csv', 'dev_data/val_rs.csv',
                       'Input directory with images.')
tf.flags.DEFINE_string('input_dir', 'dev_data/val_rs/',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './output',
                       'Output directory with images.')

tf.flags.DEFINE_float('last_b', 0.0, 'flooding.')

FLAGS = tf.flags.FLAGS
###################CUDA_VISIBLE_DEVICES=2 python3 vmi.py --output_dir temp1 --batch_size 10 #######
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
T_kern = gkern(15, 3)


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

def grad_finish(x, one_hot, i, max_iter, alpha, grad):
    return tf.less(i, max_iter)

def batch_grad(x, one_hot, i, max_iter, alpha, grad):
    x_neighbor = x + tf.random.uniform(x.shape, minval=-alpha, maxval=alpha)
    x_neighbor_2 = 1/2. * x_neighbor
    x_neighbor_4 = 1/4. * x_neighbor
    x_neighbor_8 = 1/8. * x_neighbor
    x_neighbor_16 = 1/16. * x_neighbor

    x_res = tf.concat([x_neighbor, x_neighbor_2, x_neighbor_4, x_neighbor_8, x_neighbor_16], axis=0)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x_res), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
        grad += tf.gradients(cross_entropy, x_neighbor)[0]
    i = tf.add(i, 1)
    return x, one_hot, i, max_iter, alpha, grad

def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def admix(x):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    return tf.concat([(x + FLAGS.portion * tf.gather(x, tf.random.shuffle(indices))) for _ in range(FLAGS.size)], axis=0)


def graph(x, x_ori, y, i, x_max, x_min, grad, variance):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = 10
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    x_ori = tf.cond(tf.equal(i, 0), lambda: x, lambda: x_ori)
    pre_attack = FLAGS.pre_attack
    # NI-FGSM https://arxiv.org/pdf/1908.06281.pdf
    x_nes = x + momentum * alpha * tf.sign(grad)
    # x_nes = tf.clip_by_value(x_nes, x_min, x_max)
    # x_nes = admix(x)
    # x_nes = x
    x_batch = tf.concat(
        [x_nes, x_nes/2., x_nes/4., x_nes/8., x_nes/16.], axis=0)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    pred = tf.argmax(end_points_v3['Predictions'], 1)

    # with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    #     logits_v4, end_points_v4 = inception_v4.inception_v4(
    #         input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    # pred = tf.argmax(end_points_v4['Predictions'], 1)

    # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    #     logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
    #         input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    # pred = tf.argmax(logits_resnet, 1)

    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    #     logits_Incres, end_points_IR = inception_resnet_v2.inception_resnet_v2(
    #         input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    # pred = tf.argmax(logits_Incres, 1)

    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5, axis=0)
    # one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5 * FLAGS.size, axis=0) #admix
    
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    new_grad = tf.reduce_mean(tf.split(tf.gradients(cross_entropy, x_batch)[
        0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)
    # new_grad = tf.reduce_sum(tf.split(new_grad, FLAGS.size), axis=0)  # admix
    iter = tf.constant(0)
    max_iter = tf.constant(FLAGS.number)
    # VT-FGSM: https://arxiv.org/abs/2103.15571.pdf
    # _, _, _, _, _, global_grad = tf.while_loop(grad_finish, batch_grad, [x, one_hot, iter, max_iter, eps*FLAGS.beta, tf.zeros_like(new_grad)])
    current_grad = new_grad #+ variance
    # TI-FGSM: https://arxiv.org/pdf/1904.02884.pdf
    noise = tf.nn.depthwise_conv2d(current_grad, stack_kernel, strides=[
                                   1, 1, 1, 1], padding='SAME')

    # MI-FGSM: https://arxiv.org/pdf/1710.06081.pdf
    # noise = current_grad
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise

    # variance = global_grad / (1. * FLAGS.number) - new_grad
    
    # x = tf.cond(tf.equal(i, pre_attack), lambda: x_ori, lambda: x)
    # amp = tf.cond(tf.less(i, pre_attack), lambda: 10.0, lambda: 1.0)
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, x_ori, y, i, x_max, x_min, noise, variance


def stop(x, x_ori, y, i, x_max, x_min, grad, variance):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat(
        [one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal(
        [tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width,
                            FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(
        input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                    pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape(
        (input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(
        FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

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
        #         adv_img, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
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
        x_adv, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, x_ori, y, i, x_max, x_min, tf.zeros(shape=batch_shape), tf.zeros(shape=batch_shape)])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        # s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s9.restore(sess, model_checkpoint_map['resnet_v2_101'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2_152'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])
            # s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
            dev = pd.read_csv(FLAGS.input_csv)
            idx = 0
            l2_diff = 0
            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label = load_images(
                    FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
                # diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                # l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))
                pre_ensadv_res_v2_ = sess.run(pre_ensadv_res_v2,
                                              feed_dict={adv_img: (adv_images)})
                sum_ensadv_res_v2 += (pre_ensadv_res_v2_ != True_label).sum()
                print(sum_ensadv_res_v2)

            # for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
            #     images, filenames, True_label = load_images(FLAGS.output_dir, dev, idx * FLAGS.batch_size, batch_shape)
            #     x_max1 = tf.clip_by_value(images + eps, -1.0, 1.0)
            #     x_min1 = tf.clip_by_value(images - eps, -1.0, 1.0)
            #     # images = tf.to_float(jpeg_compress(images, x_max1, x_min1)).eval()
            #     pre_ensadv_res_v2_ = sess.run(pre_ensadv_res_v2,
            #                                   feed_dict={adv_img: (images)})
            #     sum_ensadv_res_v2 += (pre_ensadv_res_v2_ != True_label).sum()
            #     print(sum_ensadv_res_v2)


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
           for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
