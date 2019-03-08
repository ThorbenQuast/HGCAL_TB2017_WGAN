import tensorflow as tf
from helpers.tf_helpers import *
from architectures.generic_network import generic_network_class

class generic_energy_regressor(generic_network_class):
  def __init__(self, batch_size, channels, trainable, reuse, scope_name="energy_regressor"):
    super(generic_energy_regressor, self).__init__(batch_size, channels, trainable, reuse, scope_name)
  
  def __call__(self, images):
    with tf.variable_scope(self.scope_name) as scope:
      if self.reuse or self.initalized:
        scope.reuse_variables()

      conv0_e, conv0w_e = conv3d(tf.expand_dims(images, -1), 1, k_d=3, k_h=3, k_w=3, d_d=1, d_h=1, d_w=1, name='e_conv0', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      e_bn0 = batch_norm(name='e_bn0', train=self.trainable)
      conv0_e = lrelu(e_bn0(conv0_e))    #12x15x7

      conv1_e, conv1w_e = conv3d(conv0_e, 16, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='e_conv1', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      e_bn1 = batch_norm(name='e_bn1', train=self.trainable)
      conv1_e = lrelu(e_bn1(conv1_e))    #10x13x6      

      conv2_e, conv2w_e = conv3d(conv1_e, 16, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='e_conv2', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      e_bn2 = batch_norm(name='e_bn2', train=self.trainable)
      conv2_e = lrelu(e_bn2(conv2_e))    #8x11x5 

      conv3_e, conv3w_e = conv3d(conv2_e, 32, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='e_conv3', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      e_bn3 = batch_norm(name='e_bn3', train=self.trainable)
      conv3_e = lrelu(e_bn3(conv3_e))    #6x9x4 

      conv4_e, conv4w_e = conv3d(conv3_e, 32, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='e_conv4', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      e_bn4 = batch_norm(name='e_bn4', train=self.trainable)
      conv4_e = lrelu(e_bn4(conv4_e))    #4x7x3 

      conv5_e, conv5w_e, conv5b_e = conv3d(conv4_e, 64, k_d=3, k_h=5, k_w=2, d_d=1, d_h=1, d_w=1, name='e_conv5', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1)  
      conv5_e = lrelu(conv5_e)    #2x3x2 

      lin_shape_e = tf.reshape(conv5_e, [self.batch_size, -1])

      h5_e, h5w_e, h5b_e = linear(lin_shape_e, 1, 'e_h5_lin', max_val=0.01, with_w=True)    
      h5_e = tf.nn.relu(h5_e)

      energy_regressor = tf.identity(h5_e, name="energy_regressor")

      self.initalized = True

      return energy_regressor

