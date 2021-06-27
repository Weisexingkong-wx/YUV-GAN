import tensorflow as tf
import tensorflow.contrib.slim as slim

import tflearn
import numpy as np
import os
import matplotlib.pylab as plt
# import cv2

import initializer as init
from utils import utils

class CloudRemovalGAN(object):

    def __init__(self, params:init.TrainingParamInitialization):

        self.img_size = params.img_size
        self.img_channel = params.img_channel
        self.alpha_channel = params.alpha_channel
        self.reflectance_channel = params.reflectance_channel

        self.batch_size = params.batch_size
        self.D_input_size = params.D_input_size
        self.G_input_size = params.G_input_size

        self.image_dir = params.image_dir
        self.test_dir = params.test_dir
        self.checkpoint_gan = params.checkpoint_gan
        self.sample_dir = params.sample_dir

        self.result_dir = params.result_dir
        self.GT_dir = params.GT_dir

        self.g_learning_rate = params.g_learning_rate
        self.d_learning_rate = params.d_learning_rate
        self.sample_size = params.sample_size
        self.decay_rate = params.decay_rate

        self.d_clip = params.d_clip         # gradient clip on the generater
        self.gan_model = params.gan_model   # valinila gan, WGAN, and LSGAN
        self.optimizer = params.optimizer

        self.Z = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.img_channel])
        self.bg = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.img_channel])
        # self.BG = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.img_channel])

        self.iter_step = params.iter_step
        self.data = utils.load_images(self.image_dir, self.img_size)
        self.test_data = utils.load_test_images(self.test_dir, self.img_size)

        print('start building GAN graphics...')
        self.build_graphics()

# define the residual block containing Relu
    def ResidualBlock(self, x,  in_channel):
        conv1 = slim.conv2d(x, in_channel, kernel_size=3, stride=1, padding='SAME',activation_fn=None)
        bn1 = tf.layers.batch_normalization(conv1, axis=-1, training=True)
        relu1 = tf.nn.relu(bn1)
        conv2 = slim.conv2d(relu1, in_channel, kernel_size=3, stride=1, padding='SAME', activation_fn=None )
        bn2 = tf.layers.batch_normalization(conv2, axis=-1, training=True)
        out = x + bn2
        return out

# define LeakyRelu activation function
    def LeakyRelu(self, x, leak=0.2, name='LeakyRelu'):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * tf.abs(x)

# define the residual block containing LeakyRelu
    def Leaky_ResidualBlock(self, x,  in_channel):
        conv1 = slim.conv2d(x, in_channel, kernel_size=3, stride=1, padding='SAME',activation_fn=None)
        bn1 = tf.layers.batch_normalization(conv1, axis=-1, training=True)
        LeakyRelu1 = self.LeakyRelu(bn1, leak=0.2)
        conv2 = slim.conv2d(LeakyRelu1, in_channel, kernel_size=3, stride=1, padding='SAME', activation_fn=None )
        bn2 = tf.layers.batch_normalization(conv2, axis=-1, training=True)
        out = x + bn2
        return out

# define the transition block，whose function is to obtain a specific number of feature maps
    def TransitionBlock(self, x,  out_channel):
        out = slim.conv2d(x, out_channel, kernel_size=1, stride=1, padding='SAME', activation_fn=None)
        out = tf.nn.relu(out)
        return out

# define the transition block containing LeakyRelu
    def Leaky_TransitionBlock(self, x, out_channel):
        out = slim.conv2d(x, out_channel, kernel_size=1, stride=1, padding='SAME', activation_fn=None)
        out = self.LeakyRelu(out, leak=0.2)
        return out

# define the upsampling block
    def upconv_blcok(self, in_maps, out_size, out_num, kernel_size=3, stride=1, method = 1):  # method = 1: the nearest interpolation

        upsample = tf.image.resize_images(in_maps, [out_size, out_size], method)
        conv = tf.contrib.slim.conv2d(upsample, out_num, kernel_size, stride, padding='SAME',
                      activation_fn=tf.nn.relu, normalizer_fn=None)
        return conv

# define the generator network
    def generator(self, Z, bg,  reuse=None):   #Z:input cloudy images; bg:background,reference cloud-free images

        if (reuse):
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('Generator_scope'):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_initializer = tf.contrib.layers.xavier_initializer(),
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                normalizer_fn=slim.batch_norm
                                ):

                Z_ = tf.image.resize_images(images=Z, size=[self.G_input_size, self.G_input_size])
                bg_ = tf.image.resize_images(images=bg, size=[self.G_input_size, self.G_input_size])

        # encoder
                with tf.variable_scope('feature_exatraction'):
                    net_a_1 = tflearn.conv_2d(Z_, 32, 3, regularizer='L2', weight_decay=0.0001)
                    net_a_1 = tf.nn.relu(net_a_1)

                    net_a_1 = self.ResidualBlock(net_a_1, 32)

                    net_a_2 = self.ResidualBlock(net_a_1, 32)

                    net_a_3 = self.ResidualBlock(net_a_2, 32)

                    net_a_4 = self.ResidualBlock(net_a_3, 32)

                    net_a_5 = self.ResidualBlock(net_a_4, 32)

        # decoder
                with tf.variable_scope('output_prediction'):

                    net_5 = self.ResidualBlock(net_a_5, 32)
                    net_5 = tf.concat([net_5, net_a_5], axis=-1)
                    net_5 = self.TransitionBlock(net_5, 32)

                    net_4 = self.ResidualBlock(net_5, 32)
                    net_4 = tf.concat([net_4, net_a_4], axis=-1)
                    net_4 = self.TransitionBlock(net_4, 32)

                    net_3 = self.ResidualBlock(net_4, 32)
                    net_3 = tf.concat([net_3, net_a_3], axis=-1)
                    net_3 = self.TransitionBlock(net_3, 32)

                    net_2 = self.ResidualBlock(net_3, 32)
                    net_2 = tf.concat([net_2, net_a_2], axis=-1)
                    net_2 = self.TransitionBlock(net_2, 32)

                    net_1 = self.ResidualBlock(net_2, 32)
                    net_1 = tf.concat([net_1, net_a_1], axis=-1)

                    output = slim.conv2d(net_1, self.img_channel, kernel_size=3, stride=1, padding='SAME',
                                     activation_fn=tf.nn.relu)

                    bg_reconstruct = slim.conv2d(output,self.img_channel, kernel_size=3, stride=1, padding='SAME',
                                             activation_fn=None)
                reconstruct_loss = tf.abs(bg_reconstruct - bg_)
                reconstruct_loss = tf.reduce_sum(reconstruct_loss)

        return bg_reconstruct, reconstruct_loss

    # define the discriminator network
    def discriminator(self, x, reuse=False):

        if (reuse):
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('Discriminator_scope'):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                normalizer_fn=slim.batch_norm,
                                # weights_regularizer=slim.l2_regularizer(0.01)
                                ):
                x = tf.image.resize_images(images=x, size=[self.D_input_size, self.D_input_size])

                net = self.Leaky_ResidualBlock(x, self.img_channel)
                net = self.Leaky_TransitionBlock(net, 32)
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                net = self.Leaky_ResidualBlock(net, 32)
                net = self.Leaky_TransitionBlock(net, 64)
                net = slim.max_pool2d(net, [2, 2], scope='pool2')

                net = self.Leaky_ResidualBlock(net, 64)
                net = self.Leaky_TransitionBlock(net, 128)
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                net = self.Leaky_ResidualBlock(net, 128)
                net = self.Leaky_TransitionBlock(net, 256)
                net = slim.max_pool2d(net, [2, 2], scope='pool4')

                net = self.Leaky_ResidualBlock(net, 256)
                net = self.Leaky_TransitionBlock(net, 512)
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                net = slim.flatten(net)  # Reducing the feature maps to a one-dimensional array

                net = slim.fully_connected(net, 512)

                D_logit = slim.fully_connected(net, 1, activation_fn=None)
                D_prob = tf.nn.sigmoid(D_logit)

        return D_logit, D_prob

    # loss functions
    def calc_loss(self, D_logits_real, D_logits_fake, G_reconstruct_loss):

        penalty =  tf.reduce_mean(G_reconstruct_loss)

        if self.gan_model is 'W_GAN': # WGAN
            D_loss = - (tf.reduce_mean(D_logits_real) - tf.reduce_mean(D_logits_fake))
            G_loss = - tf.reduce_mean(D_logits_fake)+penalty
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)

        elif self.gan_model is 'Vanilla_GAN': # Vanilla_GAN -CGAN
            D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
            D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))
            D_loss = D_loss_real + D_loss_fake
            G_loss = tf.reduce_mean(tf. nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                labels=tf.ones_like(D_logits_fake))) + penalty
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)

        else: # LS_GAN
            D_loss = 0.5 * (tf.reduce_mean((D_logits_real - 1)**2) + tf.reduce_mean(D_logits_fake**2))
            G_loss = 0.5 * tf.reduce_mean((D_logits_fake - 1)**2)+penalty
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)

        return G_loss, D_loss

    def load_historical_model(self, sess):

        # check and create model dir
        if os.path.exists(self.checkpoint_gan) is False:
            os.makedirs(self.checkpoint_gan)

        if 'checkpoint' in os.listdir(self.checkpoint_gan):
            # training from the last checkpoint
            print('loading model from the last checkpoint ...')
            saver = tf.train.Saver(
                utils.get_variables_to_restore(['Generator_scope', 'Discriminator_scope', 'G_steps', 'D_steps'],
                                               ['Adam', 'Adam_1']))
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_gan)
            saver.restore(sess, latest_checkpoint)
            print(latest_checkpoint)
            print('loading finished!')
        else:
            print('no historical gan model found, start training from scratch!')

    def run_train_loop(self):

        G_bg_reconstruct, G_reconstruct_loss  = self.generator(self.Z, self.bg)
        D_logits_real, D_prob_real = self.discriminator(self.bg)
        D_logits_fake, D_prob_fake = self.discriminator(G_bg_reconstruct, reuse=True)

        G_loss, D_loss = self.calc_loss(D_logits_real, D_logits_fake, G_reconstruct_loss)

        tvars = tf.trainable_variables()
        theta_D = [var for var in tvars if 'Discriminator_scope' in var.name]
        theta_G = [var for var in tvars if 'Generator_scope' in var.name]

        # to record the iteration number
        G_steps = tf.Variable(0, trainable=False, name='G_steps')
        D_steps = tf.Variable(0, trainable=False, name='D_steps')

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            decay_steps = 10 * self.sample_size / self.batch_size
            global_ = tf.Variable(tf.constant(0))
            update_g_learning_rate = tf.train.exponential_decay(self.g_learning_rate, global_, decay_steps,
                                                                self.decay_rate, staircase=True)
            update_d_learning_rate = tf.train.exponential_decay(self.d_learning_rate, global_, decay_steps,
                                                                self.decay_rate, staircase=False)

            if self.optimizer is 'RMSProp':
                D_solver = (tf.train.RMSPropOptimizer(learning_rate=update_d_learning_rate).minimize(D_loss, var_list=theta_D,
                                                                                              global_step=D_steps))
                G_solver = (tf.train.RMSPropOptimizer(learning_rate=update_g_learning_rate).minimize(G_loss, var_list=theta_G,
                                                                                              global_step=G_steps))

            if self.optimizer is 'Adam':
                D_solver = (tf.train.AdamOptimizer(learning_rate=update_d_learning_rate).minimize(D_loss, var_list=theta_D,
                                                                                           global_step=D_steps))
                G_solver = (tf.train.AdamOptimizer(learning_rate=update_g_learning_rate).minimize(G_loss, var_list=theta_G,
                                                                                           global_step=G_steps))

            clip_D = [p.assign(tf.clip_by_value(p, -self.d_clip, self.d_clip)) for p in theta_D]

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # create a summary writer
        merged_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('temp', sess.graph)

        # load historical model and create a saver
        self.load_historical_model(sess)
        saver = tf.train.Saver()

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        g_iter = 0
        d_iter = 0
        mm = 1  # number of G_steps per D_step

        while g_iter < self.iter_step:
            Z_mb, bg_mb = utils.img_mask_batch(self.data, self.batch_size, self.img_size)
            _, D_loss_curr, _, summary = sess.run([D_solver, D_loss, clip_D, merged_op],
                                                  feed_dict={self.Z: Z_mb, self.bg: bg_mb})
            d_iter = sess.run(D_steps)
            # write states to summary
            summary_writer.add_summary(summary, g_iter)

            # To stabilize training, we train multiple steps (mm) on G if D loss is less than a pre-defined
            # threshold, say, 0.1. We found this simple mofification on the training config greatly prevents
            # from model collapse.
            if D_loss_curr < 0.1:
                mm = mm + 5
            else:
                mm = 1

            for _ in range(mm):
                Z_mb, bg_mb = utils.img_mask_batch(self.data, self.batch_size, self.img_size)
                _, G_loss_curr, D_loss_curr = sess.run([G_solver, G_loss, D_loss],
                                                       feed_dict={self.Z: Z_mb, self.bg: bg_mb})
                g_iter = sess.run(G_steps)

                # Save the cloud-free results every 200 times
                if g_iter % 200 == 0:

                    bg_reconstruct, reconstruct_loss = sess.run([G_bg_reconstruct, G_reconstruct_loss],
                                                    feed_dict={self.Z: Z_mb, self.bg: bg_mb})

                    Z = np.array(Z_mb*255, np.uint8)
                    bg_reconstruct = np.array(bg_reconstruct*255, np.uint8)
                    bg = np.array(bg_mb*255, np.uint8)

                    for jj in range(self.batch_size):
                        i = g_iter

                        save_path = os.path.join(self.sample_dir,
                                                 '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_cloudimg.png')
                        plt.imsave(save_path, Z[jj, :, :, :])

                        save_path = os.path.join(self.sample_dir,
                                                 '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_bg_reconstruct.png')
                        plt.imsave(save_path, bg_reconstruct[jj, :, :, :])

                        save_path = os.path.join(self.sample_dir,
                                                 '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_BG.png')
                        plt.imsave(save_path, bg[jj, :, :, :])

                if g_iter % 50 == 0:
                    print('D_loss = %g, G_loss = %g' % (D_loss_curr, G_loss_curr))
                    print('g_iter = %d, d_iter = %d, n_g/d = %d' % (g_iter, d_iter, mm))

                # save model every 2000 g_iters
                if np.mod(g_iter, 2000) == 1 and g_iter > 1:
                    print('saving model to checkpoint ...')
                    saver.save(sess, os.path.join(self.checkpoint_gan, 'G_step'), global_step=G_steps)

    def build_graphics(self):
        self.G_bg_reconstruct, self.G_reconstruct_loss = self.generator(self.Z, self.bg)
        self.D_logits_real, self.D_prob_real = self.discriminator(self.bg)
        self.D_logits_fake, self.D_prob_fake = self.discriminator(self.G_bg_reconstruct, reuse=True)
        self.G_loss, self.D_loss = self.calc_loss(self.D_logits_real, self.D_logits_fake,  self.G_reconstruct_loss)

        tvars = tf.trainable_variables()
        theta_D = [var for var in tvars if 'Discriminator_scope' in var.name]
        theta_G = [var for var in tvars if 'Generator_scope' in var.name]

        # to record the iteration number
        self.G_steps = tf.Variable(0, trainable=False, name='G_steps')
        self.D_steps = tf.Variable(0, trainable=False, name='D_steps')

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            decay_steps = 10 * self.sample_size / self.batch_size
            global_ = tf.Variable(tf.constant(0))
            update_g_learning_rate = tf.train.exponential_decay(self.g_learning_rate, global_, decay_steps,
                                                                self.decay_rate, staircase=True)
            update_d_learning_rate = tf.train.exponential_decay(self.d_learning_rate, global_, decay_steps,
                                                                self.decay_rate, staircase=False)

            if self.optimizer is 'RMSProp':
                self.D_solver = (
                tf.train.RMSPropOptimizer(learning_rate=update_d_learning_rate).minimize(self.D_loss, var_list=theta_D,
                                                                                       global_step=self.D_steps))
                self.G_solver = (
                tf.train.RMSPropOptimizer(learning_rate=update_g_learning_rate).minimize(self.G_loss, var_list=theta_G,
                                                                                       global_step=self.G_steps))

            if self.optimizer is 'Adam':
                self.D_solver = (
                tf.train.AdamOptimizer(learning_rate=update_d_learning_rate).minimize(self.D_loss, var_list=theta_D,
                                                                                    global_step=self.D_steps))
                self.G_solver = (
                tf.train.AdamOptimizer(learning_rate=update_g_learning_rate).minimize(self.G_loss, var_list=theta_G,
                                                                                    global_step=self.G_steps))

            self.clip_D = [p.assign(tf.clip_by_value(p, -self.d_clip, self.d_clip)) for p in theta_D]


    def run_train(self, sess):
        Z_mb, bg_mb = utils.img_mask_batch(self.data, self.batch_size, self.img_size)
        _, D_loss_curr, _ = sess.run([self.D_solver, self.D_loss, self.clip_D],
                                     feed_dict={self.Z: Z_mb, self.bg: bg_mb})
        d_iter = sess.run(self.D_steps)

        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        mm = 1
        if D_loss_curr < 0.1:
            mm = mm + 1
        else:
            mm = 1

        for _ in range(mm):
            Z_mb, bg_mb = utils.img_mask_batch(self.data, self.batch_size, self.img_size)
            _, G_loss_curr, D_loss_curr = sess.run([self.G_solver, self.G_loss, self.D_loss],
                                                   feed_dict={self.Z: Z_mb, self.bg: bg_mb})
            g_iter = sess.run(self.G_steps)
            # save generated samples
            if g_iter % 100 == 0:
                bg_reconstruct,  reconstruct_loss = sess.run([self.G_bg_reconstruct, self.G_reconstruct_loss],
                                                feed_dict={self.Z: Z_mb, self.bg: bg_mb})
                save_path = os.path.join(self.sample_dir, '{}_bg_reconstruct.png'.format(str(g_iter).zfill(5)))
                plt.imsave(save_path, utils.plot2x2(bg_reconstruct), vmax=1, vmin=0)

                save_path = os.path.join(self.sample_dir, '{}_input.png'.format(str(g_iter).zfill(5)))
                plt.imsave(save_path, utils.plot2x2(Z_mb), vmax=1, vmin=0)

            if g_iter % 50 == 0:
                print('D_loss = %g, G_loss = %g' % (D_loss_curr, G_loss_curr))
                print('g_iter = %d, d_iter = %d, n_g/d = %d' % (g_iter, d_iter, mm))
                print()

    def run_test_loop(self):
        G_bg_reconstruct, G_reconstruct_loss = self.generator(self.Z, self.bg)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # load the detection model
        self.load_historical_model(sess)    # Loading the training model

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        for i in range(self.sample_size):
            Z_mb, bg_mb = utils.test_batch(self.test_data, i)
            bg_reconstruct,reconstruct_loss = sess.run([G_bg_reconstruct, G_reconstruct_loss],
                                           feed_dict={self.Z: Z_mb, self.bg: bg_mb})

            for jj in range(self.batch_size):
                save_path1 = os.path.join(self.GT_dir, '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '.png')
                plt.imsave(save_path1, utils.plot2x2_test(bg_mb), vmax=1, vmin=0)

                save_path = os.path.join(self.result_dir, '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '.png')
                plt.imsave(save_path,  utils.plot2x2_test(bg_reconstruct), vmax=1, vmin=0)
