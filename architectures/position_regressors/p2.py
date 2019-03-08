import tensorflow as tf
from helpers.tf_helpers import *
from architectures.generic_network import generic_network_class

class generic_position_regressor(generic_network_class):
  def __init__(self, batch_size, channels, trainable, reuse, scope_name="position_regressor"):
    super(generic_position_regressor, self).__init__(batch_size, channels, trainable, reuse, scope_name)

  def __call__(self, images):
    with tf.variable_scope(self.scope_name) as scope:
      if self.reuse or self.initalized:
        scope.reuse_variables()

      conv0_p, conv0w_p = conv3d(tf.expand_dims(images, -1), 1, k_d=3, k_h=3, k_w=3, d_d=1, d_h=1, d_w=1, name='p_conv0', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      p_bn0 = batch_norm(name='e_bn0', train=self.trainable)
      conv0_p = lrelu(p_bn0(conv0_p))    #12x15x7

      conv1_p, conv1w_p = conv3d(conv0_p, 16, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='p_conv1', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      p_bn1 = batch_norm(name='p_bn1', train=self.trainable)
      conv1_p = lrelu(p_bn1(conv1_p))    #10x13x6      

      conv2_p, conv2w_p = conv3d(conv1_p, 16, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='p_conv2', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      p_bn2 = batch_norm(name='p_bn2', train=self.trainable)
      conv2_p = lrelu(p_bn2(conv2_p))    #8x11x5 

      conv3_p, conv3w_p = conv3d(conv2_p, 32, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='p_conv3', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      p_bn3 = batch_norm(name='p_bn3', train=self.trainable)
      conv3_p = lrelu(p_bn3(conv3_p))    #6x9x4 

      conv4_p, conv4w_p = conv3d(conv3_p, 32, k_d=3, k_h=3, k_w=2, d_d=1, d_h=1, d_w=1, name='p_conv4', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)  
      p_bn4 = batch_norm(name='p_bn4', train=self.trainable)
      conv4_p = lrelu(p_bn4(conv4_p))    #4x7x3 

      conv5_p, conv5w_p, conv5b_p = conv3d(conv4_p, 64, k_d=3, k_h=5, k_w=2, d_d=1, d_h=1, d_w=1, name='p_conv5', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1)  
      conv5_p = lrelu(conv5_p)    #2x3x2 

      lin_shape_p = tf.reshape(conv5_p, [self.batch_size, -1])

      h5_p, h5w_p, h5b_p = linear(lin_shape_p, 2, 'p_h5_lin', max_val=0.01, with_w=True)    

      position_regressor = tf.identity(h5_p, name="position_regressor")

      self.initalized = True

      return position_regressor

