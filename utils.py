# coding: utf-8
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from imageio import imread, imsave
import csv


# target
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


def load_images_tar(input_dir, index, batch_shape, image_id_list, label_ori_list, label_tar_list):
    """Images for inception classifier are normalized to be in [-1, 1] interval"""
    images = np.zeros(batch_shape)
    filenames = []
    targetlabel = []
    idx = 0
    for i in range(index, min(index + batch_shape[0], 1000)):
        ImageID = image_id_list[i]
        img_path = os.path.join(input_dir, ImageID+".png")
        image = Image.open(img_path)
        # print(image.size, images.shape)
        if image.size == (224, 224):
            image = image.resize((299, 299))
        # print(image.size, images.shape)
        images[idx, ...] = np.array(image).astype(np.float) / 255.0
        filenames.append(ImageID)
        targetlabel.append(label_tar_list[i])
        idx += 1
    images = images * 2.0 - 1.0
    return images, filenames, targetlabel

def load_images_jiang(img_root, index, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    # image_size = (299, 299)

    for i in range(index, min(index + batch_shape[0], 1000)):
        # img_p = str(i) if img_root.split('/')[-1] != 'NCF' else str(i + 1)
        img_p = str(i)
        img_p = img_p + '.png'
        # image = Image.open(os.path.join(img_root, img_p)).convert('RGB').resize(image_size)
        image = Image.open(os.path.join(img_root, img_p)).convert('RGB')
        images[idx, ...] = np.array(image).astype(np.float) / 255.0
        filenames.append(img_p)
        
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames

def load_images_chen(img_root, label_root, index, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    truelabel = []
    idx = 0
    labels_f = open(label_root).readlines()
    image_size = (299, 299)

    for i in range(index, min(index + batch_shape[0], 1000)):
        # img_p = str(i) if img_root.split('/')[-1] != 'NCF' else str(i + 1)
        img_p = str(i)
        img_p = img_p + '.png'
        image = Image.open(os.path.join(img_root, img_p)).convert('RGB').resize(image_size)

        images[idx, ...] = np.array(image).astype(np.float) / 255.0
        filenames.append(img_p)
        truelabel.append(int(labels_f[i]))
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames, truelabel

def load_images(input_dir, csv_file, index, batch_shape):
    """Images for inception classifier are normalized to be in [-1, 1] interval"""
    images = np.zeros(batch_shape)
    filenames = []
    truelabel = []
    idx = 0
    for i in range(index, min(index + batch_shape[0], 1000)):
        img_obj = csv_file.loc[i]
        ImageID = img_obj['filename']
        img_path = os.path.join(input_dir, ImageID)
        image = Image.open(img_path)
        # print(image.size, images.shape)
        if image.size == (224, 224):
            image = image.resize((299, 299))
        # print(image.size, images.shape)
        images[idx, ...] = np.array(image).astype(np.float) / 255.0
        filenames.append(ImageID)
        truelabel.append(img_obj['label'])
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames, truelabel


def load_images_SSA_data(input_dir, csv_file, index, batch_shape):
    """Images for inception classifier are normalized to be in [-1, 1] interval"""
    images = np.zeros(batch_shape)
    filenames = []
    truelabel = []
    idx = 0
    for i in range(index, min(index + batch_shape[0], 1000)):
        img_obj = csv_file.loc[i]
        ImageID = img_obj['ImageId']
        img_path = os.path.join(input_dir, ImageID+'.png')
        image = Image.open(img_path)
        # print(image.size, images.shape)
        if image.size == (224, 224):
            image = image.resize((299, 299))
        # print(image.size, images.shape)
        images[idx, ...] = np.array(image).astype(np.float) / 255.0
        filenames.append(ImageID)
        truelabel.append(img_obj['TrueLabel'])
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames, truelabel


# def load_images(input_dir, batch_shape):
#     """Read png images from input directory in batches.
#     Args:
#       input_dir: input directory
#       batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
#     Yields:
#       filenames: list file names without path of each image
#         Lenght of this list could be less than batch_size, in this case only
#         first few images of the result are elements of the minibatch.
#       images: array with all images from this batch
#     """
#     images = np.zeros(batch_shape)
#     filenames = []
#     idx = 0
#     batch_size = batch_shape[0]
#     for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
#         with tf.gfile.Open(filepath, 'rb') as f:
#             image = imread(f, pilmode='RGB').astype(np.float) / 255.0
#         # Images for inception classifier are normalized to be in [-1, 1] interval.
#         images[idx, :, :, :] = image * 2.0 - 1.0
#         filenames.append(os.path.basename(filepath))
#         idx += 1
#         if idx == batch_size:
#             yield filenames, images
#             filenames = []
#             images = np.zeros(batch_shape)
#             idx = 0
#     if idx > 0:
#         yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory."""
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            image = (images[i, :, :, :] + 1.0) * 0.5
            img = Image.fromarray((image * 255).astype('uint8')).convert('RGB')
            img.save(output_dir + filename, quality=95)


def images_to_FD(input_tensor):
    """Process the image to meet the input requirements of FD"""
    ret = tf.image.resize_images(input_tensor, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ret = tf.reverse(ret, axis=[-1])  # RGB to BGR
    ret = tf.transpose(ret, [0, 3, 1, 2])
    return ret
