import tensorflow as tf
import os

#  Configuration parameters
tf.app.flags.DEFINE_integer('batch_size',1,'')
tf.app.flags.DEFINE_integer('img_size',256,'')
tf.app.flags.DEFINE_integer('D_input_size',256,'')
tf.app.flags.DEFINE_integer('G_input_size',256,'')
tf.app.flags.DEFINE_integer('img_channel',3,'')
tf.app.flags.DEFINE_integer('iter_step',20001,'')

# the setting of g_learning_rate: 1e-4 for normal training, 1e-5 for fine-tuning
tf.app.flags.DEFINE_float('g_learning_rate',1e-4,'training learning rate')
# the setting of d_learning_rate: 1e-5 for normal training, 1e-6 for fine-tuning
tf.app.flags.DEFINE_float('d_learning_rate',1e-5,'training learning rate')
tf.app.flags.DEFINE_integer('sample_size',840,'')  # Total number of training samples
tf.app.flags.DEFINE_float('decay_rate',0.85,'training learning rate') # The decay rate of the learning rate

tf.app.flags.DEFINE_string('gan_model','LSGAN','W_GAN Vanilla_GAN和LSGAN')

tf.app.flags.DEFINE_string('image_dir',r'dataset\RICE2_Landsat8',
                           'images to train GAN or CloudMatting')
tf.app.flags.DEFINE_string('checkpoint_gan',r'model\WGAN_',
                           'checkpoint path for GAN')
tf.app.flags.DEFINE_string('sample_dir',r'samples\RICE2_Landsat8',
                           'training result path for GAN or CloudMatting')

tf.app.flags.DEFINE_string('test_dir',r'dataset\RICE2_Landsat8',
                           'images to train GAN or CloudMatting')
tf.app.flags.DEFINE_string('result_dir',r'results\RICE2_Landsat8',
                           'Directory name to save the generated images,only for cloud_gereration.py')
tf.app.flags.DEFINE_string('GT_dir',r'GTs\RICE2_Landsat8',
                           'Directory name to save the generated images,only for cloud_gereration.py')

FLAGS=tf.app.flags.FLAGS