import os
import time
import tensorflow as tf
from keras.layers import LocallyConnected2D as LocallyConnected2D, UpSampling2D
import pprint
pp = pprint.PrettyPrinter()

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

flags = tf.app.flags

def save(saver, sess, checkpoint_dir, step):
  model_name = "WGAN.model"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  pp.pprint("[%s] saving the parameters after %s steps to %s..." % (time.ctime(), step, checkpoint_dir))
  saver.save(sess,
    os.path.join(checkpoint_dir, model_name),
    global_step=step)

def load(saver, sess, checkpoint_dir, loadCounter=-1):
  import re
  print(" [*] Reading checkpoints...")

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
    if loadCounter!=-1:
      ckpt_name = ckpt_name.replace(str(counter), str(loadCounter))    
      counter = loadCounter
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    pp.pprint(" [*] Success to read {}".format(ckpt_name))
    return True, counter
  else:
    pp.pprint(" [*] Failed to find a checkpoint")
    return False, 0

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm", train=True):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name
      self.train=train
  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=self.train,
                      scope=self.name)

class layer_norm(object):
  def __init__(self, train=True, name="layer_norm"):
    with tf.variable_scope(name):
      self.name = name
    self.train=train
  def __call__(self, x):
    return tf.contrib.layers.layer_norm(x,
                      trainable=self.train)



#implementation provided in Jonas' presentation at the RWTH DL meeting on 31st July 2018
def sn(W, collections=None, seed=None, return_norm=False, name='sn', num_sn_samples=1, Ip_sn=1): 
  shape = W.get_shape().as_list()
  if len(shape) == 1:
    sigma = tf.reduce_max(tf.abs(W)) 
  else:
    if len(shape) == 4:
      _W = tf.reshape(W, (-1, shape[3]))
      shape = (shape[0] * shape[1] * shape[2], shape[3])
    else:
      _W = W
    u = tf.get_variable(name=name + "_u", shape=(num_sn_samples, shape[0]), initializer=tf.random_normal_initializer, collections=collections, trainable=False)
    _u = u
    for _ in range(Ip_sn):
      _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
      _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)
    _u = tf.stop_gradient(_u)
    _v = tf.stop_gradient(_v)
    sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1)) 
    update_u_op = tf.assign(u, _u)
    with tf.control_dependencies([update_u_op]): 
      sigma = tf.identity(sigma)
    if return_norm:
      return W / sigma, sigma
    else:
      return W / sigma  



#locally connected layers in critics lead to an explosion of the EM distance
def locally_connected2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, min_val=-0.001, max_val=0.01,
       name="locallyconnected2d", no_bias=False, _padding='valid'):
  with tf.variable_scope(name):
    local_connect = LocallyConnected2D(output_dim, (k_h, k_w), input_shape=(input_.shape[1], input_.shape[2], input_.shape[3]), strides=[d_h, d_w], padding=_padding, name=name, use_bias=(not no_bias), kernel_initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
    if not no_bias:
      biases = tf.get_variable('biases', [input_.shape[1]+1-k_h, input_.shape[2]+1-k_w, output_dim], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
      
  if not no_bias:
    return tf.add(local_connect(input_), biases)
  else:
    return local_connect(input_)


def upsampling2D(input_, size=2, name="upsampling2d"):
  with tf.variable_scope(name):
    upsampling = UpSampling2D(size)
  return upsampling(input_)


def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, min_val=-0.001, max_val=0.01,
       name="conv2d", no_bias=False, bias_type="default", with_w=False, _padding='SAME', spectral_normalization=None):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
    if spectral_normalization is not None:
      w = sn(w, num_sn_samples=spectral_normalization[1], name=spectral_normalization[0])    
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=_padding)

    if not no_bias:
      if bias_type=="traditional":
        biases = tf.get_variable('biases', [output_dim], initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
        conv = tf.nn.bias_add(conv, biases)
      else:
        biases = tf.get_variable('biases', [conv.shape[1], conv.shape[2], conv.shape[3]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001))
        conv = tf.add(conv, biases)

      if with_w:
        return conv, w, biases
      else:
        return conv

    else:
      if with_w:
        return conv, w
      else:
        return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, min_val=-0.001, max_val=0.01,
       name="deconv2d", with_w=False, _padding="VALID", no_bias=False, spectral_normalization=None):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
    if spectral_normalization is not None:
      w = sn(w, num_sn_samples=spectral_normalization[1], name=spectral_normalization[0])
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding=_padding)
    if not no_bias:
      biases = tf.get_variable('biases', [deconv.shape[1], deconv.shape[2], deconv.shape[3]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001))
      deconv = tf.add(deconv, biases)

    if with_w:
      if not no_bias:
        return deconv, w, biases
      else:
        return deconv, w
    else:
      return deconv


def conv3d(input_, output_dim, 
       k_d=3, k_h=5, k_w=5, d_d=1, d_h=2, d_w=2, min_val=-0.001, max_val=0.01,
       name="conv3d", no_bias=False, bias_type="default", with_w=False, _padding='SAME', spectral_normalization=None):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
    if spectral_normalization is not None:
      w = sn(w, num_sn_samples=spectral_normalization[1], name=spectral_normalization[0])    
    conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=_padding)

    if not no_bias:
      if bias_type=="traditional":
        biases = tf.get_variable('biases', [output_dim], initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
        conv = tf.nn.bias_add(conv, biases)
      else:
        biases = tf.get_variable('biases', [conv.shape[1], conv.shape[2], conv.shape[3], conv.shape[4]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001))
        conv = tf.add(conv, biases)
      if with_w:
        return conv, w, biases
      else:
        return conv

    else:
      if with_w:
        return conv, w
      else:
        return conv

def deconv3d(input_, output_shape,
       k_d=3, k_h=5, k_w=5, d_d=1, d_h=2, d_w=2, min_val=-0.001, max_val=0.01,
       name="deconv3d", with_w=False, _padding="VALID", no_bias=False, spectral_normalization=None):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_uniform_initializer(minval=min_val, maxval=max_val))
    if spectral_normalization is not None:
      w = sn(w, num_sn_samples=spectral_normalization[1], name=spectral_normalization[0])

    deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_d, d_h, d_w, 1], padding=_padding, data_format='NDHWC')
    if not no_bias:
      biases = tf.get_variable('biases', [deconv.shape[1], deconv.shape[2], deconv.shape[3], deconv.shape[4]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001))
      deconv = tf.add(deconv, biases)

    if with_w:
      if not no_bias:
        return deconv, w, biases
      else:
        return deconv, w
    else:
      return deconv


def conv2d_kernel(input_, kernel, output_shape, 
        min_val=-0.01, max_val=0.01,
        name="conv2d_kernel", with_w=False, _padding="SAME"):
  with tf.variable_scope(name):
    deconv = tf.nn.conv3d(tf.expand_dims(input_, 0), kernel, [1, 1, 1, 1, 1], _padding)
    deconv = tf.squeeze(deconv, axis=0)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001))
    deconv = tf.add(deconv, biases)

    if with_w:
      return deconv, biases
    else:
      return deconv    


def noiseCut(x, cut, name="noiseCut"):
  return tf.multiply(x, tf.maximum(0., tf.sign(x-cut)))

   
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x, name=name)


def linear(input_, output_size, scope=None, min_val=-0.001, max_val=0.001, with_w=False, no_bias=False, spectral_normalization=None):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_uniform_initializer(minval=min_val, maxval=max_val))
    if spectral_normalization is not None:
      matrix = sn(matrix, num_sn_samples=spectral_normalization[1], name=spectral_normalization[0])    
    if no_bias==False:
      bias = tf.get_variable("bias", [output_size],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001))
      if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
      else:
        return tf.matmul(input_, matrix) + bias
    else:
      if with_w:
        return tf.matmul(input_, matrix), matrix
      else:
        return tf.matmul(input_, matrix)   

"""
wGAN implemented on top of tensorflow as described in: [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
with improvements as described in: [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf).
https://gist.github.com/mjdietzx/a8121604385ce6da251d20d018f9a6d6
"""


# define earth mover distance (wasserstein loss)
def em_loss(y_coefficients, y_pred):
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

#https://stackoverflow.com/questions/40158633/how-to-solve-nan-loss
def soft_max_cross_entropy(logits, labels):
  return -(labels*tf.log(logits + 1e-10) + (1.-labels)*tf.log(1.-logits + 1e-10))