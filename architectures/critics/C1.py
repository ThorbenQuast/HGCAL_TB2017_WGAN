import tensorflow as tf
from helpers.tf_helpers import *
from architectures.generic_network import generic_network_class

class generic_critic(generic_network_class):
  def __init__(self, batch_size, channels, trainable, reuse, scope_name="critic"):
    super(generic_critic, self).__init__(batch_size, channels, trainable, reuse, scope_name)
  
  def __call__(self, images, impactPoint, energy):
    with tf.variable_scope(self.scope_name) as scope:
      if self.reuse or self.initalized:
        scope.reuse_variables()
      conditionals_in1_c, conditionals_in1_c_w, conditionals_in1_c_b = linear(tf.concat((impactPoint, energy), axis=1), 10, 'c_condition_lin1', with_w=True, min_val=-0.1, max_val=0.1)
      conditionals_in1_c = lrelu(conditionals_in1_c)
      
      conditionals_in_c, conditionals_in_c_w, conditionals_in_c_b = linear(conditionals_in1_c, 12*15, 'c_condition_lin2', with_w=True, min_val=-0.1, max_val=0.1)
      conditionals_in_c = lrelu(conditionals_in_c)
      conditionals_in_c = tf.reshape(conditionals_in_c, [-1, 12, 15, 1]) 

      h0_c, h0w_c, h0b_c = conv2d(tf.concat((images, conditionals_in_c), axis=3), 256, k_h=5, k_w=5, d_h=1, d_w=1,  name='c_h0_conv', _padding="SAME", bias_type="traditional", with_w=True, min_val=-0.1, max_val=0.1)
      h0_c = lrelu(h0_c)

      h1_c, h1w_c, h1b_c = conv2d(h0_c, 128, k_h=3, k_w=3, d_h=1, d_w=1,  name='c_h1_conv', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      c_ln1 = layer_norm(name="c_ln1", train=self.trainable)
      h1_c = lrelu(c_ln1(h1_c))

      h2_c, h2w_c, h2b_c = conv2d(h1_c, 64, k_h=3, k_w=3, d_h=1, d_w=1,  name='c_h2_conv', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      c_ln2 = layer_norm(name="c_ln2", train=self.trainable)
      h2_c = lrelu(c_ln2(h2_c))

      h3_c, h3w_c, h3b_c = conv2d(h2_c, 32, k_h=3, k_w=3, d_h=1, d_w=1,  name='c_h3_conv', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      c_ln3 = layer_norm(name="c_ln3", train=self.trainable)
      h3_c = lrelu(c_ln3(h3_c))

      h4_c, h4w_c, h4b_c = conv2d(h3_c, 16, k_h=3, k_w=3, d_h=1, d_w=1,  name='c_h4_conv', _padding="SAME", with_w=True, min_val=-0.1, max_val=0.1, bias_type="traditional")
      c_ln4 = layer_norm(name="c_ln4", train=self.trainable)
      h4_c = lrelu(c_ln4(h4_c))

      h4_c_reshaped = tf.reshape(h4_c, [self.batch_size, -1])
      
      h5_c, h5w_c, h5b_c = linear(h4_c_reshaped, 10, 'c_h5_lin', with_w=True)    
      h5_c = lrelu(h5_c)

      h6_c, h6w_c, h6b_c = linear(h5_c, 1, 'c_h6_lin', with_w=True)
      critic = tf.identity(h6_c, name="critic")

      self.initalized = True

      return critic

