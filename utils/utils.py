
import numpy as np
import tensorflow as tf
import cv2
import glob
import tensorflow.contrib.slim as slim
from collections import OrderedDict
import os

def get_variables_to_restore(scope_to_include, suffix_to_exclude):
    """to parse which var to include and which
    var to exclude"""

    vars_to_include = []
    for scope in scope_to_include:
        vars_to_include += slim.get_variables(scope)

    vars_to_exclude = set()
    for scope in suffix_to_exclude:
        vars_to_exclude |= set(
            slim.get_variables_by_suffix(scope))

    return [v for v in vars_to_include if v not in vars_to_exclude]

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict

# 32 means for data augmentation
def data_augmentation_together(batch, bg, img_size):

    batch_size = batch.shape[0]

    # left-right flip
    if np.random.rand(1) > 0.5:
        batch = batch[:, :, ::-1, :]
        bg = bg[:, :, ::-1, :]

    # up-down flip
    if np.random.rand(1) > 0.5:
        batch = batch[:, ::-1, :, :]
        bg = bg[:, ::-1, :, :]

    # rotate 90
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=1)  # 90
            bg[id, :, :, :] = np.rot90(bg[id, :, :, :], k=1)

    # rotate 180
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=2)  # 180
            bg[id, :, :, :] = np.rot90(bg[id, :, :, :], k=2)  # 180

    # rotate 270
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=-1)  # 270
            bg[id, :, :, :] = np.rot90(bg[id, :, :, :], k=-1)  # 270

    # random crop and resize 0.5~1.0
    if np.random.rand(1) > 0.5:

        IMG_SIZE = batch.shape[1]
        scale = np.random.rand(1) * 0.5 + 0.5
        crop_height = int(scale * img_size)
        crop_width = int(scale * img_size)
        x_st = int((1 - scale) * np.random.rand(1) * (img_size - 1))
        y_st = int((1 - scale) * np.random.rand(1) * (img_size - 1))
        x_nd = x_st + crop_width
        y_nd = y_st + crop_height

        for id in range(batch_size):
            cropped_img = batch[id, y_st:y_nd, x_st:x_nd, :]
            cropped_bg = bg[id, y_st:y_nd, x_st:x_nd, :]
            batch[id, :, :, :] = cv2.resize(cropped_img, dsize=(img_size, img_size))
            bg[id, :, :, :] = cv2.resize(cropped_bg, dsize=(img_size, img_size))

    return batch, bg

def data_augmentation(batch, img_size):

    batch_size = batch.shape[0]

    # left-right flip
    if np.random.rand(1) > 0.5:
        batch = batch[:, :, ::-1, :]

    # up-down flip
    if np.random.rand(1) > 0.5:
        batch = batch[:, ::-1, :, :]

    # rotate 90
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=1)  # 90

    # rotate 180
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=2)  # 180

    # rotate 270
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=-1)  # 270

    # random crop and resize 0.5~1.0
    if np.random.rand(1) > 0.5:

        IMG_SIZE = batch.shape[1]
        scale = np.random.rand(1) * 0.5 + 0.5
        crop_height = int(scale * img_size)
        crop_width = int(scale * img_size)
        x_st = int((1 - scale) * np.random.rand(1) * (img_size - 1))
        y_st = int((1 - scale) * np.random.rand(1) * (img_size - 1))
        x_nd = x_st + crop_width
        y_nd = y_st + crop_height

        for id in range(batch_size):
            cropped_img = batch[id, y_st:y_nd, x_st:x_nd, :]
            batch[id, :, :, :] = cv2.resize(cropped_img, dsize=(img_size, img_size))

    return batch

def img_mask_batch(DATA, batch_size, img_size, with_data_augmentation=True):

    data1 = DATA['thin_cloud_images']
    n, h, w, c = data1.shape
    idx = np.random.choice(range(n), batch_size, replace=False)
    batch = data1[idx, :, :, :]

    data2 = DATA['background_images']
    bg = data2[idx, :, :, :]        # Extract the bg image corresponding to the idx number

    if with_data_augmentation is True:   # The images were selected and then enhanced
        batch, bg = data_augmentation_together(batch, bg, img_size)

    return batch, bg

def test_batch(test_data, i):
    data1 = test_data['test_thin_cloud_images']
    data2 = test_data['test_background_images']

    idx = np.array([i])
    test_batch = data1[idx, :, :, :]
    test_bg = data2[idx, :, :, :]

    return test_batch, test_bg

def plot2x2(samples):

    IMG_SIZE = samples.shape[1]

    img_grid = np.zeros((2 * IMG_SIZE, 2 * IMG_SIZE, 3),np.uint8)

    for i in range(len(samples)):
        py, px = IMG_SIZE * int(i / 2), IMG_SIZE * (i % 2)
        this_img = samples[i, :, :, :]
        this_img = np.uint8(this_img*255)
        img_grid[py:py + IMG_SIZE, px:px + IMG_SIZE, :] = this_img

    return img_grid

def plot2x2_test(samples):

    IMG_SIZE = samples.shape[1]

    img_grid = np.zeros((IMG_SIZE, IMG_SIZE, 3),np.uint8)

    for i in range(len(samples)):
        py, px = IMG_SIZE * int(i / 2), IMG_SIZE * (i % 2)
        this_img = samples[i, :, :, :]
        this_img = np.uint8(this_img*255)
        img_grid[py:py + IMG_SIZE, px:px + IMG_SIZE, :] = this_img
        img_grid = cv2.cvtColor(img_grid, cv2.COLOR_YUV2RGB)

    return img_grid

def load_images(image_dir, img_size):

    data = {
        'background_images': 0,
        'thin_cloud_images': 0,
    }

    # load images
    img_dirs = glob.glob(os.path.join(image_dir, 'label/*.png'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.resize(img, (img_size, img_size))/255.
        image_buff[i, :, :, :] = img

        i += 1
        if np.mod(i, 100) == 0:
            print('reading background images: ' + str(i) + ' / ' + str(m_tr_imgs))

    data['background_images'] = image_buff

    img_dirs = glob.glob(os.path.join(image_dir, 'cloud/*.png'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.resize(img, (img_size, img_size))/255.
        image_buff[i, :, :, :] = img

        i += 1
        if np.mod(i, 100) == 0:
            print('reading thin cloud images: ' + str(i) + ' / ' + str(m_tr_imgs))
    data['thin_cloud_images'] = image_buff

    print('loading train images done done done')

    return data

def load_test_images(test_dir, img_size):

    test_data = {
        'test_background_images': 0,
        'test_thin_cloud_images': 0,
    }

    img_dirs = glob.glob(os.path.join(test_dir, 'label/*.png'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.resize(img, (img_size, img_size))/255.
        image_buff[i, :, :, :] = img

        i += 1
        if np.mod(i, 100) == 0:
            print('reading test background images: ' + str(i) + ' / ' + str(m_tr_imgs))

    test_data['test_background_images'] = image_buff

    img_dirs = glob.glob(os.path.join(test_dir, 'cloud/*.png'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)      # COLOR_BGR2YUV, COLOR_BGR2RGB, COLOR_BGR2LAB
        img = cv2.resize(img, (img_size, img_size))/255.
        image_buff[i, :, :, :] = img

        i += 1
        if np.mod(i, 100) == 0:
            print('reading test thin cloud images: ' + str(i) + ' / ' + str(m_tr_imgs))
    test_data['test_thin_cloud_images'] = image_buff

    print('loading test images done done done')

    return test_data
