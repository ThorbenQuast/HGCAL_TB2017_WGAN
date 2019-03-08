import tensorflow as tf
from helpers.tf_helpers import *
from architectures.generic_network import generic_network_class

from helpers.config import maskDeadChannels, deadChannelList

class generic_generator(generic_network_class):
  def __init__(self, batch_size, channels, trainable, reuse, scope_name="generator"):
    super(generic_generator, self).__init__(batch_size, channels, trainable, reuse, scope_name)

  def __call__(self, z, impactPoint, energy):
    with tf.variable_scope(self.scope_name) as scope:
      if self.reuse or self.initalized:
        scope.reuse_variables()
      
      layer_deconv20x24 = []
      
      for l in range(self.channels):
        x_in, x_in_w, x_in_b = linear(tf.concat((z, impactPoint, energy), axis=1), 10, 'g_z_lin1_%s'%l, with_w=True, min_val=-0.1, max_val=0.1)
        x_in = lrelu(x_in)

        x_in2, x_in2_w, x_in2_b = linear(x_in, 3*4*16, 'g_z_lin2_%s'%l, with_w=True, min_val=-0.1, max_val=0.1)
        x_in2 = lrelu(x_in2)    
        
        x_in_reshape = tf.reshape(x_in2, [-1, 3, 4, 16]) #3x4

        h1, h1_w, h1_b = deconv2d(x_in_reshape, [self.batch_size, 6, 8, 16], k_h=3, k_w=3, d_h=2, d_w=2,  name='g_h1_deconv2_%s'%l, _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1)
        g_bn1 = batch_norm(name='g_bn1_%s'%l, train=self.trainable)
        h1 = lrelu(g_bn1(h1)) #6x8   


        h2, h2_w, h2_b = deconv2d(h1, [self.batch_size, 12, 16, 32], k_h=3, k_w=3, d_h=2, d_w=2,  name='g_h2_deconv2_%s'%l, _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1)
        g_bn2 = batch_norm(name='g_bn2_%s'%l, train=self.trainable)
        h2 = lrelu(g_bn2(h2)) #12x16  


        h3, h3_w = deconv2d(h2, [self.batch_size, 24, 32, 64], k_h=3, k_w=3, d_h=2, d_w=2,  name='g_h3_deconv2_%s'%l, _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, no_bias=True)
        g_bn3 = batch_norm(name='g_bn3_%s'%l, train=self.trainable)
        h3 = lrelu(g_bn3(h3)) #24x32  


        h4, h4_w, h4_b = conv2d(h3, 1, k_h=5, k_w=9, d_h=1, d_w=1, name='g_h4_conv_%s'%l, _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1)
        g_bn4 = batch_norm(name='g_bn4_%s'%l, train=self.trainable)
        h4 = lrelu(g_bn4(h4), name="layer_%s"%l) #20x24  

        layer_deconv20x24.append(h4)
      
      
      h5 = tf.concat([layer_deconv20x24[l] for l in range(self.channels)], axis=3)
      
      h6, h6_w, h6_b = conv2d(h5, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h6_conv', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      g_bn6 = batch_norm(name='g_bn6', train=self.trainable)
      h6 = lrelu(g_bn6(h6)) #20x24 

      h7, h7_w, h7_b = conv2d(h6, 128, k_h=5, k_w=6, d_h=1, d_w=1, name='g_h7_conv', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      g_bn7 = batch_norm(name='g_bn7', train=self.trainable)
      h7 = lrelu(g_bn7(h7)) #16x19 
 
      h8, h8_w, h8_b = conv2d(h7, 14, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h8_conv', _padding="VALID", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      g_bn8 = batch_norm(name='g_bn8', train=self.trainable)
      h8 = lrelu(g_bn8(h8)) #14x17 

      h9 = locally_connected2d(h8, self.channels, k_h=3, k_w=3, d_h=1, d_w=1,  name='g_h9_locally', _padding="VALID", min_val=-0.1, max_val=0.1)
      if not maskDeadChannels:
        generator = tf.nn.relu(h9, name="generator") 
      else:
        deadChannelMask = tf.sparse_to_dense(deadChannelList, [12, 15, self.channels], 0., default_value=1., validate_indices=False)
        generator = tf.nn.relu(tf.multiply(h9, deadChannelMask), name="generator") 


      self.initalized = True

      return generator




